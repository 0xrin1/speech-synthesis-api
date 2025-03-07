import requests
import argparse
import os
import sys
import io
import subprocess
import platform
import tempfile

def play_audio(audio_file):
    """
    Play audio using the platform's native audio player.
    Optimized for macOS, especially M-series MacBooks.
    
    Args:
        audio_file (str): Path to audio file to play
        
    Returns:
        bool: True if successful, False otherwise
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
            print(f"Unsupported platform: {system}")
            return False
            
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def speak_text(text, server_url="http://localhost:6000", output_file=None, speaker="p326", 
              play_audio_file=True, use_male_voice=True, use_high_quality=True, 
              max_gpu_memory=24, enhance_audio=True):
    """
    Send text to the speech API, save the resulting audio, and optionally play it.
    
    Args:
        text (str): The text to convert to speech
        server_url (str): URL of the speech synthesis server
        output_file (str): Path to save the WAV file
        speaker (str): Speaker ID for multi-speaker models (default: p326)
        play_audio_file (bool): Whether to play the audio after generating it
        use_male_voice (bool): Use male voice (True) or female voice (False)
        use_high_quality (bool): Use highest quality settings
        max_gpu_memory (int): Maximum GPU memory to use in GB (1-24)
        enhance_audio (bool): Apply additional GPU-based audio enhancement
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a temp file if no output file specified
        temp_file = None
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_file = temp_file.name
            temp_file.close()
        
        # Set up parameters with acceleration options
        params = {
            "text": text, 
            "speaker": speaker,
            "use_male_voice": use_male_voice,
            "use_high_quality": use_high_quality,
            "max_gpu_memory": max_gpu_memory,
            "enhance_audio": enhance_audio
        }
        
        # GET request with acceleration parameters
        response = requests.get(
            f"{server_url}/tts", 
            params=params,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return False
        
        # Save to file
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        print(f"Audio saved to {output_file}")
        
        # Play the audio if requested
        if play_audio_file:
            print("Playing audio...")
            play_audio(output_file)
            
        # Remove temp file if we created one
        if temp_file and os.path.exists(output_file):
            os.unlink(output_file)
            
        return True
        
    except Exception as e:
        print(f"Error communicating with speech API: {e}")
        # Clean up temp file if there was an error
        if temp_file and os.path.exists(output_file):
            os.unlink(output_file)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:6000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, default="../output/speech.wav", help="Output WAV file path. If not provided, a temporary file will be used.")
    parser.add_argument("--speaker", type=str, default="p326", help="Speaker ID for multi-speaker models (default: p326, deep male voice)")
    parser.add_argument("--no-play", action="store_true", help="Don't play the audio after generating it")
    parser.add_argument("--save-only", action="store_true", help="Generate audio file without playing (same as --no-play)")
    
    # GPU acceleration and quality options
    parser.add_argument("--male", dest="use_male_voice", action="store_true", default=True, help="Use male voice (default)")
    parser.add_argument("--female", dest="use_male_voice", action="store_false", help="Use female voice")
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True, help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false", help="Use lower quality for faster generation")
    parser.add_argument("--gpu-memory", dest="max_gpu_memory", type=int, default=24, help="Maximum GPU memory to use in GB (1-24, default: 24)")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True, help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false", help="Skip additional audio enhancement")
    
    args = parser.parse_args()
    
    # If no output file is specified, use a temp file
    if args.output == "../output/speech.wav" and (args.no_play or args.save_only):
        # User wants to save but didn't specify a custom path, so use the default
        output_path = args.output
        
        # Convert relative path to absolute path
        if not os.path.isabs(output_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.normpath(os.path.join(script_dir, output_path))
            
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    elif args.output != "../output/speech.wav":
        # User specified a custom output path
        output_path = args.output
        
        # Convert relative path to absolute path
        if not os.path.isabs(output_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.normpath(os.path.join(script_dir, output_path))
            
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        # No specific output needed, will use temp file
        output_path = None
    
    # Call API and save/play output with GPU acceleration options
    result = speak_text(
        args.text, 
        server_url=args.url, 
        output_file=output_path, 
        speaker=args.speaker,
        play_audio_file=not (args.no_play or args.save_only),
        use_male_voice=args.use_male_voice,
        use_high_quality=args.use_high_quality,
        max_gpu_memory=args.max_gpu_memory,
        enhance_audio=args.enhance_audio
    )
    
    # Only print path info if we're saving to a non-temporary file
    if output_path and result:
        print(f"\nAudio file saved to: {output_path}")
        if platform.system() == "Darwin":
            print("You can also play it with: afplay " + output_path)
        elif platform.system() == "Linux":
            print("You can also play it with: aplay " + output_path)
        elif platform.system() == "Windows":
            print("You can also play it with: explorer " + output_path)