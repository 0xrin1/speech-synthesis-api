"""
Base client for the Speech API.
"""

import os
import io
import requests
import logging
from typing import Dict, Any, Optional, Union, BinaryIO
import tempfile

logger = logging.getLogger(__name__)

class BaseClient:
    """
    Base client for interacting with the Speech API.
    """
    
    def __init__(self, server_url: str = "http://localhost:6000"):
        """
        Initialize the client.
        
        Args:
            server_url: URL of the speech synthesis server
        """
        self.server_url = server_url
        
    def get_speech(
        self,
        text: str,
        speaker: Optional[str] = None,
        use_male_voice: bool = True,
        use_high_quality: bool = True,
        enhance_audio: bool = True
    ) -> requests.Response:
        """
        Send text to the speech API using GET request.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID for multi-speaker models
            use_male_voice: Use male voice (True) or female voice (False)
            use_high_quality: Use highest quality settings
            enhance_audio: Apply additional GPU-based audio enhancement
            
        Returns:
            Response object from the API
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Set up parameters
            params = {
                "text": text, 
                "use_male_voice": use_male_voice,
                "use_high_quality": use_high_quality,
                "enhance_audio": enhance_audio
            }
            
            # Add optional parameters if provided
            if speaker:
                params["speaker"] = speaker
            
            # GET request with parameters
            response = requests.get(
                f"{self.server_url}/tts", 
                params=params,
                stream=True
            )
            
            if response.status_code != 200:
                error_message = f"Error: {response.status_code} - {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error communicating with speech API: {e}")
            raise
    
    def post_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        use_male_voice: bool = True,
        use_high_quality: bool = True,
        enhance_audio: bool = True
    ) -> requests.Response:
        """
        Send text to the speech API using POST request.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID for multi-speaker models
            speed: Speech speed factor (1.0 is normal)
            use_male_voice: Use male voice (True) or female voice (False)
            use_high_quality: Use highest quality settings
            enhance_audio: Apply additional GPU-based audio enhancement
            
        Returns:
            Response object from the API
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Create JSON payload
            payload = {
                "text": text,
                "use_male_voice": use_male_voice,
                "use_high_quality": use_high_quality,
                "enhance_audio": enhance_audio
            }
            
            # Add optional parameters if provided
            if voice_id:
                payload["voice_id"] = voice_id
                
            if speed != 1.0:
                payload["speed"] = speed
                
            # POST request
            response = requests.post(
                f"{self.server_url}/tts", 
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                error_message = f"Error: {response.status_code} - {response.text}"
                logger.error(error_message)
                raise Exception(error_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error communicating with speech API: {e}")
            raise
    
    def normalize_output_path(self, output_path: str) -> str:
        """
        Normalize an output path, ensuring directories exist.
        
        Args:
            output_path: The path where the output file will be saved
            
        Returns:
            Normalized absolute path
        """
        # Convert relative path to absolute path if needed
        if not os.path.isabs(output_path):
            # Make relative to current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.normpath(os.path.join(script_dir, output_path))
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return output_path
    
    def save_audio(self, response: requests.Response, output_path: str) -> str:
        """
        Save audio from a response to a file.
        
        Args:
            response: Response object from the API
            output_path: Path to save the audio file
            
        Returns:
            Path to the saved file
            
        Raises:
            Exception: If saving fails
        """
        try:
            # Normalize the output path
            normalized_path = self.normalize_output_path(output_path)
            
            # Save to file
            with open(normalized_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Audio saved to {normalized_path}")
            return normalized_path
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    def get_temp_audio_file(self, response: requests.Response) -> str:
        """
        Save audio to a temporary file.
        
        Args:
            response: Response object from the API
            
        Returns:
            Path to the temporary file
            
        Raises:
            Exception: If saving fails
        """
        try:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save to the temporary file
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            raise