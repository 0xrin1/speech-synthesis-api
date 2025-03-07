import os
import io
import subprocess
import json
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import torch
import torch.cuda
import torch.backends.cudnn
from TTS.api import TTS

# Import our custom enhancer for high-quality speech
from enhancer import process_speech, enhance_speech

# Initialize FastAPI
app = FastAPI(title="Speech Synthesis API")

def get_least_active_gpu():
    """Always use GPU device 2."""
    return 2

# Check if CUDA is available and accessible
try:
    if torch.cuda.is_available():
        # Force GPU 2
        device = "cuda:2" 
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print(f"Using CUDA device 2: {torch.cuda.get_device_name(2)}")
        
        # Verify we're actually using it
        torch.cuda.set_device(2)
        print(f"Active GPU: {torch.cuda.current_device()}")
        
        # Reserve maximum memory immediately
        with torch.no_grad():
            # Get total memory on GPU 2
            total_memory = torch.cuda.get_device_properties(2).total_memory
            # Reserve 90% of it
            reserve_size = int(0.9 * total_memory)
            # Create a tensor to hold the reservation
            dummy = torch.empty(reserve_size, dtype=torch.uint8, device="cuda:2")
            print(f"Reserved {reserve_size/1024**3:.1f}GB GPU memory")
    else:
        # Fall back to CPU if CUDA not available
        device = "cpu"
        print("CUDA not available, using CPU instead.")
        print("Speech synthesis will be much slower without GPU acceleration.")
except Exception as e:
    # Handle permission issues or other CUDA errors
    device = "cpu"
    print(f"Error accessing CUDA: {e}")
    print("Falling back to CPU mode (speech synthesis will be much slower).")
    
print(f"Using device: {device}")

# Initialize TTS model with maximal quality settings using GPU
try:
    # Determine the best models for maximum quality
    # These are the highest quality models available
    primary_model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    multi_speaker_model_name = "tts_models/en/vctk/vits"
    
    # Note: We already reserved maximum GPU memory earlier if possible
    if device.startswith("cuda"):
        print(f"Using maximum GPU memory for models")
        print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Store the dummy tensor in app state to prevent garbage collection if it exists
        if 'dummy' in locals():
            app.state.reserved_memory = dummy
            print("Memory reservation is active")
    else:
        print("Running in CPU mode - no GPU memory to reserve")
    
    # Default to Tacotron2-DDC with HiFiGAN vocoder (ultra-high quality female voice)
    print(f"Loading primary ultra-high-quality model: {primary_model_name}...")
    tts = TTS(model_name=primary_model_name)
    
    # For male voice mode, load VCTK VITS model with p311 speaker (best deep male voice)
    print(f"Loading multi-speaker model for deep male voice: {multi_speaker_model_name}...")
    multi_speaker_tts = TTS(model_name=multi_speaker_model_name)
    
    # Configure for highest possible quality output
    # p311, p326, and p364 are excellent deep male voices based on testing
    default_speaker = "p311"  # Very deep, clear male voice
    
    # Set device and optimize for maximum quality
    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
        # Maximize GPU performance
        torch.set_grad_enabled(False)  # Disable gradients for inference
        torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster math
        
        # Increase precision and quality settings
        print("Configuring models for maximum quality...")
        
        # Move both models to GPU with ultra-high quality settings
        tts.synthesizer.to(device)
        multi_speaker_tts.synthesizer.to(device)
        
        # Set high-precision computation where possible
        if hasattr(tts.synthesizer, 'forward_attention'):
            tts.synthesizer.forward_attention = True  # Better alignment for smoother speech
        
        # Increase sampling for better quality (higher values = more computation/quality)
        if hasattr(tts.synthesizer, 'max_decoder_steps'):
            tts.synthesizer.max_decoder_steps = 2000  # Default is much lower, this allows for better quality
            
        # Voice clone settings - push more GPU resources into voice quality
        if hasattr(multi_speaker_tts.synthesizer, 'encoder_sample_rate'):
            multi_speaker_tts.synthesizer.encoder_sample_rate = 48000  # Higher sample rate
    
    # Check how much GPU memory we're using
    print(f"Memory usage after loading both models: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"High-quality TTS models loaded successfully on {device}!")
    print(f"Using default male voice: {default_speaker}")
    
    # Test that we can access speakers
    if hasattr(multi_speaker_tts, "speakers") and multi_speaker_tts.speakers:
        male_speakers = [s for s in multi_speaker_tts.speakers if isinstance(s, str) and s.startswith('p3')]
        print(f"Available male speakers: {len(male_speakers)}")
        print(f"Selected deep male voice: {default_speaker}")
    
except Exception as e:
    print(f"Error loading primary model: {e}")
    # Fallback to just the VITS model if loading both fails
    try:
        print("Falling back to VITS model only...")
        tts = TTS(model_name="tts_models/en/vctk/vits")
        multi_speaker_tts = tts  # Use the same model for both
        default_speaker = "p326"  # Default male voice
        
        # Set device manually if possible
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
            tts.synthesizer.to(device)
        print(f"Fallback TTS model loaded successfully on {device}!")
    except Exception as e:
        print(f"Error loading fallback TTS model: {e}")
        tts = None
        multi_speaker_tts = None

@app.get("/")
def read_root():
    """Root endpoint showing API info."""
    # GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(current_device),
            "gpu_index": current_device,
            "total_gpus": torch.cuda.device_count(),
            "memory_allocated": f"{torch.cuda.memory_allocated(current_device) / 1024**2:.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(current_device) / 1024**2:.2f} MB",
            "memory_total": f"{torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB"
        }
    
    # Model information
    models_info = {}
    if tts is not None:
        models_info["female_voice"] = {
            "name": getattr(tts, "model_name", "tacotron2-DDC"),
            "type": "Single speaker, high-quality female voice",
            "vocoder": getattr(tts, "vocoder_name", "hifigan"),
            "quality": "Very high"
        }
    
    if multi_speaker_tts is not None:
        # Get some sample male speakers
        male_speakers = []
        if hasattr(multi_speaker_tts, "speakers") and multi_speaker_tts.speakers:
            male_speakers = [s for s in multi_speaker_tts.speakers if isinstance(s, str) and s.startswith('p3')][:5]
        
        models_info["male_voice"] = {
            "name": getattr(multi_speaker_tts, "model_name", "vits"),
            "type": "Multi-speaker with male voice options",
            "default_speaker": default_speaker,
            "sample_male_speakers": male_speakers,
            "total_speakers": len(multi_speaker_tts.speakers) if hasattr(multi_speaker_tts, "speakers") else 0
        }
    
    # API usage examples
    examples = {
        "female_voice": "/tts?text=This is high quality female voice synthesis.&use_male_voice=false",
        "male_voice": "/tts?text=This is a deep male voice using neural speech synthesis.",
        "custom_male": f"/tts?text=This is a custom male voice.&speaker={default_speaker}",
        "post_request": {
            "url": "/tts",
            "method": "POST",
            "body": {
                "text": "This is an example of the POST endpoint with more options.",
                "voice_id": default_speaker,
                "speed": 1.0,
                "use_high_quality": True,
                "use_male_voice": True
            }
        }
    }
    
    return {
        "name": "High-Quality Neural Speech Synthesis API",
        "description": "API for converting text to speech using state-of-the-art neural models with GPU acceleration",
        "endpoints": {
            "GET /tts": "Convert text to speech with query parameters",
            "POST /tts": "Convert text to speech with JSON body"
        },
        "parameters": {
            "text": "Text to convert to speech (required)",
            "speaker": f"Speaker ID for multi-speaker models (default: {default_speaker}, only used with male voice)",
            "use_high_quality": "Use highest quality settings (default: true)",
            "use_male_voice": "Use male voice (true) or female voice (false), default: true",
            "speed": "Speech speed factor (POST only, default: 1.0)"
        },
        "device": device,
        "models_loaded": tts is not None and multi_speaker_tts is not None,
        "models": models_info,
        "gpu_info": gpu_info,
        "examples": examples
    }

@app.get("/tts")
def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    speaker: str = Query(default_speaker, description="Speaker ID for multi-speaker models"),
    use_high_quality: bool = Query(True, description="Use highest quality settings"),
    use_male_voice: bool = Query(True, description="Use male voice (True) or female voice (False)"),
    max_gpu_memory: int = Query(24, description="Maximum GPU memory to use in GB (always max)"),
    enhance_audio: bool = Query(True, description="Apply additional GPU-based audio enhancement")
):
    """Convert text to speech using GET request with ultra-high quality settings."""
    if tts is None or multi_speaker_tts is None:
        raise HTTPException(status_code=500, detail="TTS models failed to load")
    
    try:
        # Select the appropriate model based on voice preference
        if use_male_voice:
            # Use VITS multi-speaker model with male voice
            model_to_use = multi_speaker_tts
            # Set up kwargs for TTS with speaker
            kwargs = {"speaker": speaker if speaker else default_speaker}
        else:
            # Use high-quality Tacotron2-DDC model (female voice)
            model_to_use = tts
            # No speaker parameter for single-speaker model
            kwargs = {}
        
        # Process GPU memory before generation
        if torch.cuda.is_available():
            # Clear cache to maximize available memory
            torch.cuda.empty_cache()
            print("Using maximum quality settings with GPU device 2")
        
        # Use our enhanced speech processor for maximum quality
        # This takes advantage of the GPU and segments the text for better results
        print(f"Processing speech with enhanced quality settings...")
        wav = process_speech(
            model=model_to_use, 
            text=text, 
            kwargs=kwargs, 
            device=device
        )
            
        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Apply additional audio enhancement if requested
        if enhance_audio and torch.cuda.is_available() and use_high_quality:
            print("Applying additional audio enhancement...")
            # Our custom enhancer applies spectral processing on the GPU
            # to further improve voice quality
            sample_rate = getattr(model_to_use.synthesizer, 'output_sample_rate', 22050)
            wav = enhance_speech(wav, sample_rate=sample_rate, device=device)
            
        # Process GPU memory after generation
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory after generation: {current_memory:.2f} GB")
        
        # Convert numpy array to bytes
        wav_bytes = io.BytesIO()
        model_to_use.synthesizer.save_wav(wav, wav_bytes)
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
    voice_id: Optional[str] = default_speaker  # Default speaker ID for male voice model
    speed: Optional[float] = 1.0
    use_high_quality: Optional[bool] = True
    use_male_voice: Optional[bool] = True  # Default to male voice
    max_gpu_memory: Optional[int] = 24  # Maximum GPU memory to use in GB (always max)
    enhance_audio: Optional[bool] = True  # Apply additional GPU-based audio enhancement

@app.post("/tts")
def text_to_speech_post(request: TextToSpeechRequest):
    """Convert text to speech using POST request with ultra-high quality settings."""
    if tts is None or multi_speaker_tts is None:
        raise HTTPException(status_code=500, detail="TTS models failed to load")
    
    try:
        # Select the appropriate model based on voice preference
        if request.use_male_voice:
            # Use VITS multi-speaker model with male voice
            model_to_use = multi_speaker_tts
            # Set up kwargs with speaker
            chosen_speaker = request.voice_id if request.voice_id else default_speaker
            # Verify the speaker exists
            if hasattr(model_to_use, "speakers"):
                if chosen_speaker in model_to_use.speakers:
                    kwargs = {"speaker": chosen_speaker}
                else:
                    # Fall back to default male speaker
                    kwargs = {"speaker": default_speaker}
            else:
                kwargs = {}
        else:
            # Use high-quality Tacotron2-DDC model (female voice)
            model_to_use = tts
            # No speaker parameter for single-speaker model
            kwargs = {}
        
        # Apply speed adjustment if provided
        if request.speed != 1.0 and hasattr(model_to_use.synthesizer, 'length_scale'):
            # Convert speed to length_scale (inverse relationship)
            kwargs['length_scale'] = 1.0 / request.speed
            
        # Process GPU memory before generation
        if torch.cuda.is_available():
            # Clear cache to maximize available memory
            torch.cuda.empty_cache()
            print("Using maximum quality settings with GPU device 2")
                    
        # Use our enhanced speech processor for maximum quality
        print(f"Processing speech with enhanced quality settings...")
        wav = process_speech(
            model=model_to_use, 
            text=request.text, 
            kwargs=kwargs, 
            device=device
        )
            
        # Free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Apply additional audio enhancement if requested
        if request.enhance_audio and torch.cuda.is_available() and request.use_high_quality:
            print("Applying additional audio enhancement...")
            sample_rate = getattr(model_to_use.synthesizer, 'output_sample_rate', 22050)
            wav = enhance_speech(wav, sample_rate=sample_rate, device=device)
            
        # Process GPU memory after generation
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory after generation: {current_memory:.2f} GB")
        
        # Convert numpy array to bytes
        wav_bytes = io.BytesIO()
        model_to_use.synthesizer.save_wav(wav, wav_bytes)
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