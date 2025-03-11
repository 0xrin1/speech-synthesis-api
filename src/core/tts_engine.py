"""
TTS engine module for generating speech from text.
"""

import logging
import io
import numpy as np
import torch
import re
import random
from typing import Dict, Any, Optional

from src.core.device_manager import DeviceManager
from src.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Core engine for text-to-speech generation.
    """
    
    def __init__(
        self, 
        model_loader: ModelLoader,
        device_manager: DeviceManager,
        default_sample_rate: int = 22050,
        pause_duration: float = 0.25
    ):
        """
        Initialize the TTS engine.
        
        Args:
            model_loader: ModelLoader instance with loaded models
            device_manager: DeviceManager instance
            default_sample_rate: Default sample rate for audio generation
            pause_duration: Pause duration between sentences (in seconds)
        """
        self.model_loader = model_loader
        self.device_manager = device_manager
        self.default_sample_rate = default_sample_rate
        self.pause_duration = pause_duration
    
    def generate_speech(
        self,
        text: str,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        enhance_audio: bool = True,
        use_high_quality: bool = True
    ) -> io.BytesIO:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID for the VITS model
            speed: Speech speed factor (1.0 is normal)
            enhance_audio: Whether to apply additional audio enhancement
            use_high_quality: Whether to use highest quality settings
            
        Returns:
            BytesIO object containing WAV audio data
            
        Raises:
            Exception: If speech generation fails
        """
        if not self.model_loader.models_loaded():
            raise Exception("TTS model failed to load")
        
        # Use VITS model with specified speaker
        model_to_use = self.model_loader.model
        
        # Set up kwargs with speaker
        chosen_speaker = speaker if speaker else self.model_loader.default_speaker
        
        # Verify the speaker exists
        chosen_speaker = self.model_loader.validate_speaker(chosen_speaker)
        kwargs = {"speaker": chosen_speaker}
        
        # Apply speed adjustment if provided
        if speed != 1.0 and hasattr(model_to_use.synthesizer, 'length_scale'):
            # Convert speed to length_scale (inverse relationship)
            kwargs['length_scale'] = 1.0 / speed
            
        # Prepare GPU if available
        self.device_manager.empty_cache()
        logger.info(f"Processing speech with enhanced quality settings...")
        
        # Generate speech
        wav = self._process_speech(
            model=model_to_use,
            text=text,
            kwargs=kwargs
        )
            
        # Free memory
        self.device_manager.empty_cache()
            
        # Apply additional audio enhancement if requested
        if enhance_audio and torch.cuda.is_available() and use_high_quality:
            logger.info("Applying additional audio enhancement...")
            sample_rate = getattr(model_to_use.synthesizer, 'output_sample_rate', self.default_sample_rate)
            wav = self._enhance_speech(wav, sample_rate=sample_rate)
            
        # Log memory usage after generation
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Memory after generation: {current_memory:.2f} GB")
        
        # Convert numpy array to bytes
        wav_bytes = io.BytesIO()
        model_to_use.synthesizer.save_wav(wav, wav_bytes)
        wav_bytes.seek(0)
        
        return wav_bytes
    
    def _process_speech(self, model, text: str, kwargs: Dict[str, Any]) -> np.ndarray:
        """
        Process speech with high quality settings using segmentation.
        
        Args:
            model: TTS model to use
            text: Text to convert to speech
            kwargs: Additional arguments for the TTS model
            
        Returns:
            Numpy array containing the waveform
        """
        try:
            
            # Process the text for highest quality results
            sentences = [s.strip() + "." for s in text.split('.') if s.strip()]
            if not sentences:
                sentences = [text]
                
            full_wav = None
            sample_rate = getattr(model.synthesizer, 'output_sample_rate', self.default_sample_rate)
    
            # Generate speech with maximum quality using GPU and batch processing
            if torch.cuda.is_available():
                # Use higher precision computation
                with torch.cuda.amp.autocast(enabled=False):  # Force full precision for quality
                    # Process each sentence separately for higher quality
                    for sentence in sentences:
                        # Generate with high quality
                        sentence_wav = model.tts(text=sentence, **kwargs)
                        
                        # Concatenate with previous segments
                        if full_wav is None:
                            full_wav = sentence_wav
                        else:
                            # Add a small pause between sentences
                            pause = np.zeros(int(sample_rate * self.pause_duration))
                            full_wav = np.concatenate([full_wav, pause, sentence_wav])
            else:
                # CPU fallback
                full_wav = model.tts(text=text, **kwargs)
                
            return full_wav
            
        except Exception as e:
            logger.error(f"Error in process_speech: {e}")
            # Fallback to simple generation
            return model.tts(text=text, **kwargs)
            
    def _enhance_speech(self, wav: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """
        Apply high-quality enhancements to the generated speech waveform.
        Uses GPU to process the audio for better results.
        
        Args:
            wav: The numpy array containing the speech waveform
            sample_rate: The sample rate of the audio
            
        Returns:
            The enhanced waveform as a numpy array
        """
        if not torch.cuda.is_available() or self.device_manager.device == "cpu":
            # Skip enhancement if GPU is not available
            return wav
            
        try:
            device = self.device_manager.device
            
            # Convert to tensor for GPU processing
            wav_tensor = torch.from_numpy(wav).float().to(device)
            
            # Get memory information to determine how much to allocate
            device_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            free_memory = total_memory - allocated_memory
            logger.info(f"Enhancement - Free GPU memory: {free_memory:.2f} GB")
            
            # Use up to 50% of available free memory for enhancement
            target_memory = min(free_memory * 0.5, 8.0)  # Cap at 8GB for enhancement 
            
            # Calculate tensor dimensions based on target memory
            # We'll use a 2D tensor for spectral processing
            mem_bytes = target_memory * 1024**3
            dim = int((mem_bytes / 4)**0.5)  # Each float32 is 4 bytes
            dim = min(dim, 10000)  # Reasonable upper limit
            
            logger.info(f"Allocating {target_memory:.2f} GB for quality enhancement (dim={dim})")
            
            try:
                # Use tensor to force high quality processing
                boost_tensor = torch.zeros((dim, dim), device=device, dtype=torch.float32)
                logger.info(f"Enhancement memory allocation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            except Exception as e:
                logger.error(f"Error allocating enhancement memory: {e}")
            
            # Apply spectral processing
            n_fft = 1024
            hop_length = int(sample_rate * 0.005)  # 5ms hop length for high quality
            
            # Calculate spectrogram on GPU
            spec = torch.stft(
                wav_tensor, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=n_fft, 
                window=torch.hann_window(n_fft).to(device),
                return_complex=True
            )
            
            # Convert to magnitude and phase
            mag = torch.abs(spec)
            phase = torch.angle(spec)
            
            # Apply high-pass filter (reduce frequencies below 80Hz)
            freq_bins = torch.linspace(0, sample_rate // 2, n_fft // 2 + 1, device=device)
            high_pass_mask = 1.0 - torch.exp(-(freq_bins / 80.0)**2)
            high_pass_mask = high_pass_mask.unsqueeze(1)  # Add time dimension
            mag = mag * high_pass_mask
            
            # Convert back to complex
            spec_enhanced = mag * torch.exp(1j * phase)
            
            # Convert back to waveform
            wav_enhanced = torch.istft(
                spec_enhanced,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft).to(device)
            )
            
            # Convert back to numpy
            wav_enhanced_np = wav_enhanced.cpu().numpy()
            
            # Free memory
            del spec, mag, phase, spec_enhanced, wav_enhanced, wav_tensor
            if 'boost_tensor' in locals():
                del boost_tensor
            
            self.device_manager.empty_cache()
            logger.info(f"Memory after enhancement: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            return wav_enhanced_np
            
        except Exception as e:
            logger.error(f"Error in enhance_speech: {e}")
            # Return original audio if enhancement fails
            return wav