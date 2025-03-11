import requests
import argparse
import os
import sys
from pydub import AudioSegment
from pydub.playback import play
import io

def speak_text(text, server_url="http://localhost:8000", output_file=None, play_audio=True, speaker="p326",
             use_high_quality=True, max_gpu_memory=24, enhance_audio=True):
    """
    Send text to the speech API and either save or play the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        server_url (str): URL of the speech synthesis server
        output_file (str, optional): Path to save the WAV file. If None, don't save.
        play_audio (bool): Whether to play the audio immediately
        speaker (str): Speaker ID for the VITS model (default: p326)
        use_high_quality (bool): Use highest quality settings
        max_gpu_memory (int): Maximum GPU memory to use in GB (1-24)
        enhance_audio (bool): Apply additional GPU-based audio enhancement
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set up parameters with GPU acceleration options
        params = {
            "text": text, 
            "speaker": speaker,
            "use_high_quality": use_high_quality,
            "max_gpu_memory": max_gpu_memory,
            "enhance_audio": enhance_audio
        }
        
        # GET request with parameters
        response = requests.get(
            f"{server_url}/tts", 
            params=params,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return False
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Audio saved to {output_file}")
        
        # Play audio if requested
        if play_audio:
            try:
                # Load the audio bytes
                audio_bytes = io.BytesIO(response.content)
                audio = AudioSegment.from_wav(audio_bytes)
                print("Playing audio...")
                play(audio)
            except Exception as e:
                print(f"Error playing audio: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error communicating with speech API: {e}")
        return False

def post_speak_text(text, voice_id=None, speed=1.0, server_url="http://localhost:8000", 
                  output_file=None, play_audio=True, use_high_quality=True,
                  max_gpu_memory=24, enhance_audio=True):
    """
    Send text to the speech API using POST with more options and either save or play the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str, optional): ID of the voice to use
        speed (float): Speed factor (1.0 is normal)
        server_url (str): URL of the speech synthesis server
        output_file (str, optional): Path to save the WAV file. If None, don't save.
        play_audio (bool): Whether to play the audio immediately
        use_high_quality (bool): Use highest quality settings
        max_gpu_memory (int): Maximum GPU memory to use in GB (1-24)
        enhance_audio (bool): Apply additional GPU-based audio enhancement
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create JSON payload with GPU acceleration options
        payload = {
            "text": text,
            "speed": speed,
            "use_high_quality": use_high_quality,
            "max_gpu_memory": max_gpu_memory,
            "enhance_audio": enhance_audio
        }
        
        if voice_id:
            payload["voice_id"] = voice_id
            
        # POST request
        response = requests.post(
            f"{server_url}/tts", 
            json=payload,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return False
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Audio saved to {output_file}")
        
        # Play audio if requested
        if play_audio:
            try:
                # Load the audio bytes
                audio_bytes = io.BytesIO(response.content)
                audio = AudioSegment.from_wav(audio_bytes)
                print("Playing audio...")
                play(audio)
            except Exception as e:
                print(f"Error playing audio: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error communicating with speech API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:6000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, default="../output/speech.wav", help="Output WAV file path")
    parser.add_argument("--no-play", action="store_true", help="Don't play the audio")
    parser.add_argument("--voice", type=str, help="Voice ID to use (if supported)")
    parser.add_argument("--speaker", type=str, default="p326", help="Speaker ID for multi-speaker models (default: p326, deep male voice)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor (1.0 is normal)")
    
    # GPU acceleration and quality options
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True, help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false", help="Use lower quality for faster generation")
    parser.add_argument("--gpu-memory", dest="max_gpu_memory", type=int, default=24, help="Maximum GPU memory to use in GB (1-24, default: 24)")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True, help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false", help="Skip additional audio enhancement")
    
    args = parser.parse_args()
    
    # Process output path if specified
    output_path = None
    if args.output:
        # Convert relative path to absolute path if needed
        output_path = args.output
        if not os.path.isabs(output_path):
            # Make relative to current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.normpath(os.path.join(script_dir, output_path))
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if args.voice or args.speed != 1.0:
        # Use POST endpoint for advanced features
        post_speak_text(
            args.text,
            voice_id=args.voice if args.voice else args.speaker,
            speed=args.speed,
            server_url=args.url,
            output_file=output_path,
            play_audio=not args.no_play,
            use_high_quality=args.use_high_quality,
            max_gpu_memory=args.max_gpu_memory,
            enhance_audio=args.enhance_audio
        )
    else:
        # Use GET endpoint
        speak_text(
            args.text,
            server_url=args.url,
            output_file=output_path,
            play_audio=not args.no_play,
            speaker=args.speaker,
            use_high_quality=args.use_high_quality,
            max_gpu_memory=args.max_gpu_memory,
            enhance_audio=args.enhance_audio
        )