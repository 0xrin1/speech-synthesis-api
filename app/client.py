import requests
import argparse
import os
import sys
from pydub import AudioSegment
from pydub.playback import play
import io

def speak_text(text, server_url="http://localhost:8000", output_file=None, play_audio=True):
    """
    Send text to the speech API and either save or play the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        server_url (str): URL of the speech synthesis server
        output_file (str, optional): Path to save the WAV file. If None, don't save.
        play_audio (bool): Whether to play the audio immediately
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Simple GET request
        response = requests.get(
            f"{server_url}/tts", 
            params={"text": text},
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

def post_speak_text(text, voice_id=None, speed=1.0, server_url="http://localhost:8000", output_file=None, play_audio=True):
    """
    Send text to the speech API using POST with more options and either save or play the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str, optional): ID of the voice to use
        speed (float): Speed factor (1.0 is normal)
        server_url (str): URL of the speech synthesis server
        output_file (str, optional): Path to save the WAV file. If None, don't save.
        play_audio (bool): Whether to play the audio immediately
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create JSON payload
        payload = {
            "text": text,
            "speed": speed
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
    parser = argparse.ArgumentParser(description="Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, help="Output WAV file path")
    parser.add_argument("--no-play", action="store_true", help="Don't play the audio")
    parser.add_argument("--voice", type=str, help="Voice ID to use (if supported)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor (1.0 is normal)")
    
    args = parser.parse_args()
    
    if args.voice or args.speed != 1.0:
        # Use POST endpoint for advanced features
        post_speak_text(
            args.text,
            voice_id=args.voice,
            speed=args.speed,
            server_url=args.url,
            output_file=args.output,
            play_audio=not args.no_play
        )
    else:
        # Use simpler GET endpoint
        speak_text(
            args.text,
            server_url=args.url,
            output_file=args.output,
            play_audio=not args.no_play
        )