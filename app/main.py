import os
import io
import subprocess
import json
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import torch
import torch.cuda
import torch.backends.cudnn
from TTS.api import TTS

# Initialize FastAPI
app = FastAPI(title="Speech Synthesis API")

def get_least_active_gpu():
    """Find the least active GPU by utilization percentage."""
    if not torch.cuda.is_available():
        return None
    
    try:
        # Get GPU info from nvidia-smi in JSON format
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the CSV output
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                idx, util, mem = line.split(',')
                gpus.append({
                    'index': int(idx.strip()),
                    'utilization': float(util.strip()),
                    'memory_used': float(mem.strip())
                })
        
        # Find the least utilized GPU (by GPU utilization first, then by memory usage)
        if gpus:
            least_active = min(gpus, key=lambda x: (x['utilization'], x['memory_used']))
            return least_active['index']
    except Exception as e:
        print(f"Error finding least active GPU: {e}")
    
    # Fallback to first GPU if we can't determine the least active
    return 0 if torch.cuda.device_count() > 0 else None

# Check if CUDA is available and select least active GPU
if torch.cuda.is_available():
    gpu_id = get_least_active_gpu()
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cuda:0"
    print(f"CUDA available with {torch.cuda.device_count()} GPUs")
    print(f"Selected GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    device = "cpu"
print(f"Using device: {device}")

# Initialize TTS model with highest quality settings
try:
    # Using Coqui TTS with the VITS model - highest quality
    # VITS is one of the best neural TTS models, perfect for utilizing GPU acceleration
    tts = TTS(model_name="tts_models/en/vctk/vits")
    
    # Configure for high quality output with a deep male voice
    # p326 is a particularly good male voice based on testing
    default_speaker = "p326"  # Deep male voice
    
    # Set device manually if possible - use entire GPU for maximum quality
    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
        # Pin memory for faster inference
        torch.set_grad_enabled(False)  # Disable gradients for inference
        torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
        
        # Move model to GPU with optimizations
        tts.synthesizer.to(device)
        
        # Set batch size to 1 for highest quality per sample
        if hasattr(tts.synthesizer, 'batch_size'):
            tts.synthesizer.batch_size = 1
    
    print(f"High-quality TTS model loaded successfully on {device}!")
    print(f"Using default male voice: {default_speaker}")
    
except Exception as e:
    print(f"Error loading TTS model: {e}")
    # Fallback to a different model if the first one fails
    try:
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        # Set device manually if possible
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
            tts.synthesizer.to(device)
        print(f"Fallback TTS model loaded successfully on {device}!")
    except Exception as e:
        print(f"Error loading fallback TTS model: {e}")
        tts = None

@app.get("/")
def read_root():
    """Root endpoint showing API info."""
    gpu_info = {}
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(current_device),
            "gpu_index": current_device,
            "total_gpus": torch.cuda.device_count(),
            "memory_allocated": f"{torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB"
        }
    
    return {
        "name": "Speech Synthesis API",
        "description": "API for converting text to speech using state-of-the-art models",
        "endpoints": {
            "GET /tts": "Convert text to speech with query parameters",
            "POST /tts": "Convert text to speech with JSON body"
        },
        "device": device,
        "model_loaded": tts is not None,
        "gpu_info": gpu_info
    }

@app.get("/tts")
def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    speaker: str = Query(default_speaker, description="Speaker ID for multi-speaker models"),
    use_high_quality: bool = Query(True, description="Use highest quality settings")
):
    """Convert text to speech using GET request."""
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model failed to load")
    
    try:
        # Set up kwargs for TTS
        kwargs = {}
        
        # Use the specified speaker or fall back to our default male voice
        if hasattr(tts, "speakers"):
            kwargs["speaker"] = speaker if speaker else default_speaker
        
        # High quality settings - leveraging the GPU for maximum quality
        if use_high_quality:
            if hasattr(tts, 'synthesizer'):
                # Configure TTS for highest quality
                if hasattr(tts.synthesizer, 'length_scale'):
                    kwargs['length_scale'] = 1.0  # Normal speed for clarity
                    
                # Apply voice enhancing post-processing if available
                if hasattr(tts.synthesizer, 'do_post_enhancement'):
                    tts.synthesizer.do_post_enhancement = True
        
        # Generate speech with maximum quality using GPU
        if torch.cuda.is_available():
            # Use proper autocast for newer PyTorch versions
            with torch.amp.autocast('cuda', enabled=True):
                wav = tts.tts(text=text, **kwargs)
        else:
            wav = tts.tts(text=text, **kwargs)
        
        # Convert numpy array to bytes
        wav_bytes = io.BytesIO()
        tts.synthesizer.save_wav(wav, wav_bytes)
        wav_bytes.seek(0)
        
        # Return audio file
        return StreamingResponse(
            wav_bytes, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = default_speaker  # Default to the male voice
    speed: Optional[float] = 1.0
    use_high_quality: Optional[bool] = True

@app.post("/tts")
def text_to_speech_post(request: TextToSpeechRequest):
    """Convert text to speech using POST request with more options."""
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model failed to load")
    
    try:
        # Set up kwargs for high quality speech generation
        kwargs = {}
        
        # Use specified voice or default to our male voice
        if hasattr(tts, "speakers"):
            # Use requested voice or fall back to deep male voice (p326)
            chosen_speaker = request.voice_id if request.voice_id else default_speaker  # default_speaker = p326
            # Verify the speaker exists
            if chosen_speaker in tts.speakers:
                kwargs["speaker"] = chosen_speaker
            else:
                kwargs["speaker"] = default_speaker
        
        # Apply speed adjustment if provided (VITS supports this directly)
        if request.speed != 1.0 and hasattr(tts.synthesizer, 'length_scale'):
            # Convert speed to length_scale (inverse relationship)
            kwargs['length_scale'] = 1.0 / request.speed
        
        # High quality settings - leveraging the GPU
        if request.use_high_quality:
            if hasattr(tts, 'synthesizer'):
                # Apply voice enhancing post-processing if available
                if hasattr(tts.synthesizer, 'do_post_enhancement'):
                    tts.synthesizer.do_post_enhancement = True
        
        # Generate speech with maximum quality using GPU acceleration
        if torch.cuda.is_available():
            # Use proper autocast for newer PyTorch versions
            with torch.amp.autocast('cuda', enabled=True):
                wav = tts.tts(text=request.text, **kwargs)
        else:
            wav = tts.tts(text=request.text, **kwargs)
        
        # Convert numpy array to bytes
        wav_bytes = io.BytesIO()
        tts.synthesizer.save_wav(wav, wav_bytes)
        wav_bytes.seek(0)
        
        # Return audio file
        return StreamingResponse(
            wav_bytes, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=speech.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Text-to-Speech API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)