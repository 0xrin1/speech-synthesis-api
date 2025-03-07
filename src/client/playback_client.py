"""
Enhanced playback client for the Speech API.
"""

import os
import logging
import subprocess
import platform
from typing import Optional
import argparse

from src.client.base_client import BaseClient

logger = logging.getLogger(__name__)

class PlaybackClient(BaseClient):
    """
    Client for generating and playing speech from the API.
    """
    
    def play_audio(self, audio_file: str) -> bool:
        """
        Play audio using the platform's native audio player.
        Optimized for macOS, especially M-series MacBooks.
        
        Args:
            audio_file: Path to audio file to play
            
        Returns:
            True if successful, False otherwise
        """
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                # Use afplay which is native to macOS and optimized for Apple Silicon
                subprocess.run(["afplay", audio_file], check=True)
            elif system == "Linux":
                # Try to use aplay on Linux
                subprocess.run(["aplay", audio_file], check=True)
            elif system == "Windows":
                # Use PowerShell's System.Media.SoundPlayer on Windows
                powershell_cmd = f'(New-Object System.Media.SoundPlayer "{audio_file}").PlaySync();'
                subprocess.run(["powershell", "-Command", powershell_cmd], check=True)
            else:
                logger.warning(f"Unsupported platform: {system}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def speak_text(
        self,
        text: str,
        output_file: Optional[str] = None,
        speaker: Optional[str] = None,
        play_audio_file: bool = True,
        use_male_voice: bool = True,
        use_high_quality: bool = True,
        enhance_audio: bool = True,
        cleanup_temp: bool = True
    ) -> bool:
        """
        Send text to the speech API, save the resulting audio, and optionally play it.
        
        Args:
            text: The text to convert to speech
            output_file: Path to save the WAV file (if None, will use a temp file)
            speaker: Speaker ID for multi-speaker models
            play_audio_file: Whether to play the audio after generating it
            use_male_voice: Use male voice (True) or female voice (False)
            use_high_quality: Use highest quality settings
            enhance_audio: Apply additional GPU-based audio enhancement
            cleanup_temp: Whether to clean up temporary files
            
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
            
            # Determine whether to use a temp file or save to specified path
            using_temp = output_file is None
            file_path = self.get_temp_audio_file(response) if using_temp else self.save_audio(response, output_file)
            
            # Play the audio if requested
            if play_audio_file:
                logger.info("Playing audio...")
                self.play_audio(file_path)
                
            # If we created a temp file, clean it up unless requested not to
            if using_temp and cleanup_temp and os.path.exists(file_path):
                os.unlink(file_path)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in speak_text: {e}")
            return False


def main():
    """Command-line interface for the playback client."""
    parser = argparse.ArgumentParser(description="Enhanced Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:6000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, help="Output WAV file path. If not provided, a temporary file will be used.")
    parser.add_argument("--speaker", type=str, default="p326", help="Speaker ID for multi-speaker models (default: p326, deep male voice)")
    parser.add_argument("--no-play", action="store_true", help="Don't play the audio after generating it")
    parser.add_argument("--save-only", action="store_true", help="Generate audio file without playing (same as --no-play)")
    
    # Voice and quality options
    parser.add_argument("--male", dest="use_male_voice", action="store_true", default=True, help="Use male voice (default)")
    parser.add_argument("--female", dest="use_male_voice", action="store_false", help="Use female voice")
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True, help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false", help="Use lower quality for faster generation")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True, help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false", help="Skip additional audio enhancement")
    
    args = parser.parse_args()
    
    # Determine output path
    output_path = None
    if args.output or args.no_play or args.save_only:
        # User wants to save the file
        output_path = args.output or "../../output/speech.wav"
    
    # Create client and speak text
    client = PlaybackClient(server_url=args.url)
    result = client.speak_text(
        text=args.text,
        output_file=output_path,
        speaker=args.speaker,
        play_audio_file=not (args.no_play or args.save_only),
        use_male_voice=args.use_male_voice,
        use_high_quality=args.use_high_quality,
        enhance_audio=args.enhance_audio
    )
    
    # Print path info if we saved to a file
    if output_path and result:
        print(f"\nAudio file saved to: {os.path.abspath(output_path)}")
        system = platform.system()
        if system == "Darwin":
            print(f"You can also play it with: afplay {os.path.abspath(output_path)}")
        elif system == "Linux":
            print(f"You can also play it with: aplay {os.path.abspath(output_path)}")
        elif system == "Windows":
            print(f"You can also play it with: explorer {os.path.abspath(output_path)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()