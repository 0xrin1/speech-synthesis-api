"""
Tests for the TTS engine.
"""

import unittest
import sys
import os
import io
from unittest.mock import MagicMock, patch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.device_manager import DeviceManager
from src.core.model_loader import ModelLoader
from src.core.tts_engine import TTSEngine


class TestTTSEngine(unittest.TestCase):
    """Test cases for the TTS engine."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.device_manager = MagicMock(spec=DeviceManager)
        self.device_manager.device = "cpu"
        self.device_manager.empty_cache.return_value = None
        
        self.model_loader = MagicMock(spec=ModelLoader)
        self.model_loader.models_loaded.return_value = True
        self.model_loader.default_speaker = "p311"
        self.model_loader.validate_speaker.return_value = "p311"
        
        # Create mock models
        self.primary_model = MagicMock()
        self.primary_model.tts.return_value = b'dummy_audio_data'
        self.primary_model.synthesizer = MagicMock()
        self.primary_model.synthesizer.output_sample_rate = 22050
        self.primary_model.synthesizer.save_wav = MagicMock()
        
        self.multi_speaker_model = MagicMock()
        self.multi_speaker_model.tts.return_value = b'dummy_audio_data'
        self.multi_speaker_model.synthesizer = MagicMock()
        self.multi_speaker_model.synthesizer.output_sample_rate = 22050
        self.multi_speaker_model.synthesizer.save_wav = MagicMock()
        
        # Assign models to loader
        self.model_loader.primary_model = self.primary_model
        self.model_loader.multi_speaker_model = self.multi_speaker_model
        
        # Create engine
        self.engine = TTSEngine(
            model_loader=self.model_loader,
            device_manager=self.device_manager
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.model_loader, self.model_loader)
        self.assertEqual(self.engine.device_manager, self.device_manager)
        self.assertEqual(self.engine.default_sample_rate, 22050)
        self.assertEqual(self.engine.pause_duration, 0.25)
    
    def test_generate_speech_male_voice(self):
        """Test speech generation with male voice."""
        # Configure save_wav to write to BytesIO buffer
        def mock_save_wav(audio_data, file_obj):
            file_obj.write(b'mock_wav_data')
            
        self.multi_speaker_model.synthesizer.save_wav.side_effect = mock_save_wav
        
        # Generate speech
        result = self.engine.generate_speech(
            text="Hello world",
            use_male_voice=True,
            speaker="p311"
        )
        
        # Check that the correct model was used
        self.model_loader.multi_speaker_model.tts.assert_called_once()
        self.assertEqual(result.getvalue(), b'mock_wav_data')
    
    def test_generate_speech_female_voice(self):
        """Test speech generation with female voice."""
        # Configure save_wav to write to BytesIO buffer
        def mock_save_wav(audio_data, file_obj):
            file_obj.write(b'mock_wav_data')
            
        self.primary_model.synthesizer.save_wav.side_effect = mock_save_wav
        
        # Generate speech
        result = self.engine.generate_speech(
            text="Hello world",
            use_male_voice=False
        )
        
        # Check that the correct model was used
        self.model_loader.primary_model.tts.assert_called_once()
        self.assertEqual(result.getvalue(), b'mock_wav_data')
    
    def test_models_not_loaded(self):
        """Test behavior when models are not loaded."""
        # Simulate models not loaded
        self.model_loader.models_loaded.return_value = False
        
        # Attempt to generate speech
        with self.assertRaises(Exception) as context:
            self.engine.generate_speech(text="Hello world")
        
        self.assertIn("TTS models failed to load", str(context.exception))


if __name__ == '__main__':
    unittest.main()