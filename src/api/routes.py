"""
API routes for the Speech API.
"""

import logging
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import TextToSpeechRequest, APIInfo
from src.core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

router = APIRouter()

class TTSRouter:
    """
    Router for Text-to-Speech endpoints.
    """
    
    def __init__(self, tts_engine: TTSEngine, default_speaker: str):
        """
        Initialize the TTS router.
        
        Args:
            tts_engine: TTSEngine instance
            default_speaker: Default speaker ID for multi-speaker models
        """
        self.tts_engine = tts_engine
        self.default_speaker = default_speaker
        self.setup_routes()
    
    def setup_routes(self):
        """Register all routes with the router."""
        router.add_api_route("/", self.read_root, methods=["GET"])
        router.add_api_route("/tts", self.text_to_speech, methods=["GET"])
        router.add_api_route("/tts", self.text_to_speech_post, methods=["POST"])
        self.router = router
    
    async def read_root(self):
        """Root endpoint showing API info."""
        # Get GPU information
        gpu_info = self.tts_engine.device_manager.get_memory_info()
        
        # Get model information
        models_info = self.tts_engine.model_loader.get_models_info()
        
        # Get some sample speakers for examples
        speakers = self.tts_engine.model_loader.get_available_speakers()[:5]
        
        # API usage examples
        examples = {
            "default_voice": "/tts?text=This is speech synthesis using the default speaker.",
            "custom_voice": f"/tts?text=This is a custom voice.&speaker={self.default_speaker}",
            "post_request": {
                "url": "/tts",
                "method": "POST",
                "body": {
                    "text": "This is an example of the POST endpoint with more options.",
                    "voice_id": self.default_speaker,
                    "speed": 1.0,
                    "use_high_quality": True
                }
            }
        }
        
        return APIInfo(
            name="High-Quality Neural Speech Synthesis API",
            description="API for converting text to speech using state-of-the-art VITS model with GPU acceleration",
            endpoints={
                "GET /tts": "Convert text to speech with query parameters",
                "POST /tts": "Convert text to speech with JSON body"
            },
            parameters={
                "text": "Text to convert to speech (required)",
                "speaker": f"Speaker ID for the VITS model (default: {self.default_speaker})",
                "use_high_quality": "Use highest quality settings (default: true)",
                "speed": "Speech speed factor (POST only, default: 1.0)"
            },
            device=self.tts_engine.device_manager.device,
            models_loaded=self.tts_engine.model_loader.models_loaded(),
            models=models_info,
            gpu_info=gpu_info,
            examples=examples
        )
    
    async def text_to_speech(
        self,
        text: str = Query(..., description="Text to convert to speech"),
        speaker: str = Query(None, description="Speaker ID for the VITS model"),
        use_high_quality: bool = Query(True, description="Use highest quality settings"),
        enhance_audio: bool = Query(True, description="Apply additional GPU-based audio enhancement")
    ):
        """Convert text to speech using GET request with ultra-high quality settings."""
        try:
            # Use default speaker if none provided
            speaker_to_use = speaker if speaker else self.default_speaker
            
            wav_bytes = self.tts_engine.generate_speech(
                text=text,
                speaker=speaker_to_use,
                enhance_audio=enhance_audio,
                use_high_quality=use_high_quality
            )
            
            return StreamingResponse(
                wav_bytes, 
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=speech.wav"}
            )
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
    
    async def text_to_speech_post(self, request: TextToSpeechRequest):
        """Convert text to speech using POST request with ultra-high quality settings."""
        try:
            speaker_to_use = request.voice_id if request.voice_id else self.default_speaker
            
            wav_bytes = self.tts_engine.generate_speech(
                text=request.text,
                speaker=speaker_to_use,
                speed=request.speed,
                enhance_audio=request.enhance_audio,
                use_high_quality=request.use_high_quality
            )
            
            return StreamingResponse(
                wav_bytes, 
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=speech.wav"}
            )
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")