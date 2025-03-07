"""
CLI client for the Speech API.
"""

import os
import logging
import argparse
from typing import Optional

from src.client.base_client import BaseClient

logger = logging.getLogger(__name__)


class CLIClient(BaseClient):
    """
    Command-line client for interacting with the Speech API.
    """
    
    def speak_text(
        self,
        text: str,
        output_file: str,
        speaker: Optional[str] = None,
        use_male_voice: bool = True,
        use_high_quality: bool = True,
        enhance_audio: bool = True
    ) -> bool:
        """
        Send text to the speech API using GET request and save the audio.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the WAV file
            speaker: Speaker ID for multi-speaker models
            use_male_voice: Use male voice (True) or female voice (False)
            use_high_quality: Use highest quality settings
            enhance_audio: Apply additional GPU-based audio enhancement
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get speech from API
            response = self.get_speech(
                text=text,
                speaker=speaker,
                use_male_voice=use_male_voice,
                use_high_quality=use_high_quality,
                enhance_audio=enhance_audio
            )
            
            # Save to file
            self.save_audio(response, output_file)
            return True
            
        except Exception as e:
            logger.error(f"Error in speak_text: {e}")
            return False
    
    def advanced_speak_text(
        self,
        text: str,
        output_file: str,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        use_male_voice: bool = True,
        use_high_quality: bool = True,
        enhance_audio: bool = True
    ) -> bool:
        """
        Send text to the speech API using POST request with advanced options.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save the WAV file
            voice_id: Voice ID for multi-speaker models
            speed: Speech speed factor (1.0 is normal)
            use_male_voice: Use male voice (True) or female voice (False)
            use_high_quality: Use highest quality settings
            enhance_audio: Apply additional GPU-based audio enhancement
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get speech from API with POST
            response = self.post_speech(
                text=text,
                voice_id=voice_id,
                speed=speed,
                use_male_voice=use_male_voice,
                use_high_quality=use_high_quality,
                enhance_audio=enhance_audio
            )
            
            # Save to file
            self.save_audio(response, output_file)
            return True
            
        except Exception as e:
            logger.error(f"Error in advanced_speak_text: {e}")
            return False


def main():
    """Command-line interface for the CLI client."""
    parser = argparse.ArgumentParser(description="Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:6000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, default="../../output/speech.wav", help="Output WAV file path")
    parser.add_argument("--voice", type=str, help="Voice ID to use (uses POST endpoint)")
    parser.add_argument("--speaker", type=str, default="p326", help="Speaker ID for GET endpoint (default: p326)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor (1.0 is normal, uses POST endpoint)")
    
    # Voice and quality options
    parser.add_argument("--male", dest="use_male_voice", action="store_true", default=True, help="Use male voice (default)")
    parser.add_argument("--female", dest="use_male_voice", action="store_false", help="Use female voice")
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True, help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false", help="Use lower quality for faster generation")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True, help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false", help="Skip additional audio enhancement")
    
    args = parser.parse_args()
    
    # Create client
    client = CLIClient(server_url=args.url)
    
    # Use advanced options if specified
    if args.voice or args.speed != 1.0:
        # Use POST endpoint for advanced features
        client.advanced_speak_text(
            text=args.text,
            output_file=args.output,
            voice_id=args.voice if args.voice else args.speaker,
            speed=args.speed,
            use_male_voice=args.use_male_voice,
            use_high_quality=args.use_high_quality,
            enhance_audio=args.enhance_audio
        )
    else:
        # Use GET endpoint
        client.speak_text(
            text=args.text,
            output_file=args.output,
            speaker=args.speaker,
            use_male_voice=args.use_male_voice,
            use_high_quality=args.use_high_quality,
            enhance_audio=args.enhance_audio
        )
    
    print(f"Audio saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()