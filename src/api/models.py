"""
API models for the Speech API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TextToSpeechRequest(BaseModel):
    """Request model for POST /tts endpoint"""
    text: str = Field(..., description="Text to convert to speech")
    voice_id: Optional[str] = Field(None, description="Speaker ID for multi-speaker models")
    speed: Optional[float] = Field(1.0, description="Speech speed factor (1.0 is normal)")
    use_high_quality: Optional[bool] = Field(True, description="Use highest quality settings")
    use_male_voice: Optional[bool] = Field(True, description="Use male voice (True) or female voice (False)")
    enhance_audio: Optional[bool] = Field(True, description="Apply additional audio enhancement")
    jamaican_accent: Optional[bool] = Field(True, description="Transform text to approximate Jamaican accent")


class APIInfo(BaseModel):
    """Model for API information response"""
    name: str
    description: str
    endpoints: dict
    parameters: dict
    device: str
    models_loaded: bool
    models: dict
    gpu_info: dict
    examples: dict