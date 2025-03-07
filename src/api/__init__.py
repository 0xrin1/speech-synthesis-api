"""API module for the Speech API."""

from src.api.routes import router, TTSRouter
from src.api.models import TextToSpeechRequest, APIInfo

__all__ = ['router', 'TTSRouter', 'TextToSpeechRequest', 'APIInfo']