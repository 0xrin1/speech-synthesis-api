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
        primary_model_name: str,
        multi_speaker_model_name: str,
        default_speaker: str,
        fallback_speaker: str,
        device_manager: DeviceManager
    ):
        """
        Initialize the model loader.
        
        Args:
            primary_model_name: Name of the primary TTS model (usually female voice)
            multi_speaker_model_name: Name of the multi-speaker model (usually for male voices)
            default_speaker: Default speaker ID for multi-speaker model
            fallback_speaker: Fallback speaker ID if default is unavailable
            device_manager: DeviceManager instance
        """
        self.primary_model_name = primary_model_name
        self.multi_speaker_model_name = multi_speaker_model_name
        self.default_speaker = default_speaker
        self.fallback_speaker = fallback_speaker
        self.device_manager = device_manager
        
        # Will be populated after loading
        self.primary_model = None
        self.multi_speaker_model = None
        self.available_speakers = []
        
    def load_models(self) -> Tuple[bool, str]:
        """
        Load the TTS models.
        
        Returns:
            Tuple of (success_flag, error_message)
        """
        try:
            # Ensure device is set up
            if self.device_manager.device is None:
                self.device_manager.setup()
                
            device = self.device_manager.device
            logger.info(f"Loading models on device: {device}")
            
            # Load primary model (female voice)
            logger.info(f"Loading primary ultra-high-quality model: {self.primary_model_name}...")
            self.primary_model = TTS(model_name=self.primary_model_name)
            
            # Load multi-speaker model (male voices)
            logger.info(f"Loading multi-speaker model: {self.multi_speaker_model_name}...")
            self.multi_speaker_model = TTS(model_name=self.multi_speaker_model_name)
            
            # Configure for maximum quality
            self._configure_models(device)
            
            # Get available speakers
            if hasattr(self.multi_speaker_model, "speakers") and self.multi_speaker_model.speakers:
                self.available_speakers = self.multi_speaker_model.speakers
                male_speakers = [s for s in self.available_speakers if isinstance(s, str) and s.startswith('p3')]
                logger.info(f"Available male speakers: {len(male_speakers)}")
                logger.info(f"Selected deep male voice: {self.default_speaker}")
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error loading primary models: {e}")
            
            # Try fallback to just the multi-speaker model
            try:
                logger.warning("Falling back to VITS model only...")
                fallback_model = TTS(model_name=self.multi_speaker_model_name)
                
                # Use the same model for both
                self.primary_model = fallback_model
                self.multi_speaker_model = fallback_model
                
                # Set device manually if possible
                if hasattr(fallback_model, 'synthesizer') and hasattr(fallback_model.synthesizer, 'to'):
                    fallback_model.synthesizer.to(device)
                
                logger.info(f"Fallback TTS model loaded successfully on {device}!")
                
                # Get available speakers
                if hasattr(fallback_model, "speakers") and fallback_model.speakers:
                    self.available_speakers = fallback_model.speakers
                
                return True, f"Warning: Using fallback model due to: {str(e)}"
                
            except Exception as fallback_error:
                logger.error(f"Error loading fallback TTS model: {fallback_error}")
                return False, f"Failed to load models: {str(e)}. Fallback also failed: {str(fallback_error)}"
    
    def _configure_models(self, device: str):
        """
        Configure loaded models for maximum quality.
        
        Args:
            device: Device to move models to (e.g., 'cuda:0', 'cpu')
        """
        # Configure GPU optimization
        self.device_manager.optimize_for_inference()
        
        # Check if we can move the model to the device
        if hasattr(self.primary_model, 'synthesizer') and hasattr(self.primary_model.synthesizer, 'to'):
            logger.info("Configuring models for maximum quality...")
            
            # Move both models to GPU with ultra-high quality settings
            self.primary_model.synthesizer.to(device)
            self.multi_speaker_model.synthesizer.to(device)
            
            # Set high-precision computation where possible
            if hasattr(self.primary_model.synthesizer, 'forward_attention'):
                self.primary_model.synthesizer.forward_attention = True  # Better alignment for smoother speech
            
            # Increase sampling for better quality (higher values = more computation/quality)
            if hasattr(self.primary_model.synthesizer, 'max_decoder_steps'):
                self.primary_model.synthesizer.max_decoder_steps = 2000  # Default is much lower
                
            # Voice clone settings - push more GPU resources into voice quality
            if hasattr(self.multi_speaker_model.synthesizer, 'encoder_sample_rate'):
                self.multi_speaker_model.synthesizer.encoder_sample_rate = 48000  # Higher sample rate
        
        # Log memory usage after loading
        mem_allocated = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"Memory usage after loading both models: {mem_allocated:.2f} MB")
    
    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        models_info = {}
        
        if self.primary_model is not None:
            models_info["female_voice"] = {
                "name": getattr(self.primary_model, "model_name", "tacotron2-DDC"),
                "type": "Single speaker, high-quality female voice",
                "vocoder": getattr(self.primary_model, "vocoder_name", "hifigan"),
                "quality": "Very high"
            }
        
        if self.multi_speaker_model is not None:
            # Get some sample male speakers
            male_speakers = []
            if hasattr(self.multi_speaker_model, "speakers") and self.multi_speaker_model.speakers:
                male_speakers = [s for s in self.multi_speaker_model.speakers 
                                if isinstance(s, str) and s.startswith('p3')][:5]
            
            models_info["male_voice"] = {
                "name": getattr(self.multi_speaker_model, "model_name", "vits"),
                "type": "Multi-speaker with male voice options",
                "default_speaker": self.default_speaker,
                "sample_male_speakers": male_speakers,
                "total_speakers": len(self.multi_speaker_model.speakers) 
                                 if hasattr(self.multi_speaker_model, "speakers") else 0
            }
        
        return models_info
    
    def get_male_speakers(self) -> List[str]:
        """
        Get a list of available male speakers.
        
        Returns:
            List of male speaker IDs
        """
        if hasattr(self.multi_speaker_model, "speakers") and self.multi_speaker_model.speakers:
            return [s for s in self.multi_speaker_model.speakers if isinstance(s, str) and s.startswith('p3')]
        return []
    
    def validate_speaker(self, speaker: str) -> str:
        """
        Validate if a speaker exists and return a valid speaker ID.
        
        Args:
            speaker: Speaker ID to validate
            
        Returns:
            Valid speaker ID (default if provided is invalid)
        """
        if hasattr(self.multi_speaker_model, "speakers"):
            if speaker in self.multi_speaker_model.speakers:
                return speaker
        
        # Return default if invalid
        return self.default_speaker
    
    def models_loaded(self) -> bool:
        """
        Check if models are loaded successfully.
        
        Returns:
            True if models are loaded, False otherwise
        """
        return self.primary_model is not None and self.multi_speaker_model is not None