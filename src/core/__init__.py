"""Core module for the Speech API."""

from src.core.device_manager import DeviceManager
from src.core.model_loader import ModelLoader
from src.core.tts_engine import TTSEngine

__all__ = ['DeviceManager', 'ModelLoader', 'TTSEngine']