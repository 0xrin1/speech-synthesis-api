"""
Model loader module for loading and configuring TTS models.
"""

import logging
from typing import Dict, Tuple, Optional, Any, List
import torch
from TTS.api import TTS
from src.core.device_manager import DeviceManager

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles loading and configuration of TTS models.
    """
    
    def __init__(
        self,
        tts_model_name: str,
        default_speaker: str,
        fallback_speaker: str,
        device_manager: DeviceManager
    ):
        """
        Initialize the model loader.
        
        Args:
            tts_model_name: Name of the TTS model to use
            default_speaker: Default speaker ID for the model
            fallback_speaker: Fallback speaker ID if default is unavailable
            device_manager: DeviceManager instance
        """
        self.tts_model_name = tts_model_name
        self.default_speaker = default_speaker
        self.fallback_speaker = fallback_speaker
        self.device_manager = device_manager
        
        # Will be populated after loading
        self.model = None
        self.available_speakers = []
        
    def load_models(self) -> Tuple[bool, str]:
        """
        Load the TTS model.
        
        Returns:
            Tuple of (success_flag, error_message)
        """
        try:
            # Ensure device is set up
            if self.device_manager.device is None:
                self.device_manager.setup()
                
            device = self.device_manager.device
            logger.info(f"Loading model on device: {device}")
            
            # Load VITS model
            logger.info(f"Loading VITS model: {self.tts_model_name}...")
            self.model = TTS(model_name=self.tts_model_name)
            
            # Configure for maximum quality
            self._configure_model(device)
            
            # Get available speakers
            if hasattr(self.model, "speakers") and self.model.speakers:
                self.available_speakers = self.model.speakers
                speakers = [s for s in self.available_speakers if isinstance(s, str) and s.startswith('p3')]
                logger.info(f"Available speakers: {len(speakers)}")
                logger.info(f"Selected default voice: {self.default_speaker}")
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error loading TTS model: {e}")
            return False, f"Failed to load model: {str(e)}"
    
    def _configure_model(self, device: str):
        """
        Configure loaded model for maximum quality.
        
        Args:
            device: Device to move model to (e.g., 'cuda:0', 'cpu')
        """
        # Configure GPU optimization
        self.device_manager.optimize_for_inference()
        
        # Check if we can move the model to the device
        if hasattr(self.model, 'synthesizer') and hasattr(self.model.synthesizer, 'to'):
            logger.info("Configuring model for maximum quality...")
            
            # Move model to GPU with ultra-high quality settings
            self.model.synthesizer.to(device)
            
            # Increase sampling for better quality (higher values = more computation/quality)
            if hasattr(self.model.synthesizer, 'max_decoder_steps'):
                self.model.synthesizer.max_decoder_steps = 2000  # Default is much lower
                
            # Voice clone settings - push more GPU resources into voice quality
            if hasattr(self.model.synthesizer, 'encoder_sample_rate'):
                self.model.synthesizer.encoder_sample_rate = 48000  # Higher sample rate
        
        # Log memory usage after loading
        mem_allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"Memory usage after loading model: {mem_allocated:.2f} MB")
    
    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        models_info = {}
        
        if self.model is not None:
            # Get some sample speakers
            sample_speakers = []
            if hasattr(self.model, "speakers") and self.model.speakers:
                sample_speakers = [s for s in self.model.speakers 
                                if isinstance(s, str) and s.startswith('p3')][:5]
            
            models_info["vits_model"] = {
                "name": getattr(self.model, "model_name", "vits"),
                "type": "Multi-speaker VITS model",
                "default_speaker": self.default_speaker,
                "sample_speakers": sample_speakers,
                "total_speakers": len(self.model.speakers) 
                                 if hasattr(self.model, "speakers") else 0
            }
        
        return models_info
    
    def get_available_speakers(self) -> List[str]:
        """
        Get a list of available speakers.
        
        Returns:
            List of speaker IDs
        """
        if hasattr(self.model, "speakers") and self.model.speakers:
            return [s for s in self.model.speakers if isinstance(s, str) and s.startswith('p3')]
        return []
    
    def validate_speaker(self, speaker: str) -> str:
        """
        Validate if a speaker exists and return a valid speaker ID.
        
        Args:
            speaker: Speaker ID to validate
            
        Returns:
            Valid speaker ID (default if provided is invalid)
        """
        if hasattr(self.model, "speakers"):
            if speaker in self.model.speakers:
                return speaker
        
        # Return default if invalid
        return self.default_speaker
    
    def models_loaded(self) -> bool:
        """
        Check if model is loaded successfully.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None