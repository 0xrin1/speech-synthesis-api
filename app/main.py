import os
import io
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import torch
from TTS.api import TTS

# Initialize FastAPI
app = FastAPI(title="Speech Synthesis API")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize TTS model
try:
    # Using Coqui TTS with the VITS model
    tts = TTS(model_name="tts_models/en/vctk/vits")
    # Set device manually if possible
    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
        tts.synthesizer.to(device)
    print(f"TTS model loaded successfully on {device}!")
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

class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    speed: Optional[float] = 1.0

@app.get("/")
def read_root():
    """Root endpoint showing API info."""
    return {
        "name": "Speech Synthesis API",
        "description": "API for converting text to speech using state-of-the-art models",
        "endpoints": {
            "GET /tts": "Convert text to speech with query parameters",
            "POST /tts": "Convert text to speech with JSON body"
        },
        "device": device,
        "model_loaded": tts is not None
    }

@app.get("/tts")
def text_to_speech(text: str = Query(..., description="Text to convert to speech")):
    """Convert text to speech using GET request."""
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model failed to load")
    
    try:
        # Generate speech
        wav = tts.tts(text=text)
        
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

@app.post("/tts")
def text_to_speech_post(request: TextToSpeechRequest):
    """Convert text to speech using POST request with more options."""
    if tts is None:
        raise HTTPException(status_code=500, detail="TTS model failed to load")
    
    try:
        # Generate speech with voice_id if provided
        kwargs = {}
        if request.voice_id and hasattr(tts, "speakers") and request.voice_id in tts.speakers:
            kwargs["speaker"] = request.voice_id
        
        # Generate speech
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