import requests
import argparse
import os
import sys
import io

def speak_text(text, server_url="http://localhost:6000", output_file=None, speaker="p335"):
    """
    Send text to the speech API and save the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        server_url (str): URL of the speech synthesis server
        output_file (str): Path to save the WAV file
        speaker (str): Speaker ID for multi-speaker models (default: p335)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Simple GET request
        response = requests.get(
            f"{server_url}/tts", 
            params={"text": text, "speaker": speaker},
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
        return True
        
    except Exception as e:
        print(f"Error communicating with speech API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Speech Synthesis API Client")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("--url", type=str, default="http://localhost:6000", help="URL of the speech API server")
    parser.add_argument("--output", "-o", type=str, default="../output/speech.wav", help="Output WAV file path")
    parser.add_argument("--speaker", type=str, default="p335", help="Speaker ID for multi-speaker models (default: p335)")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute path if needed
    output_path = args.output
    if not os.path.isabs(output_path):
        # Make relative to current script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.normpath(os.path.join(script_dir, output_path))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Call API and save output
    speak_text(args.text, server_url=args.url, output_file=output_path, speaker=args.speaker)
    
    # Print path to the generated audio file
    print(f"\nAudio file generated: {output_path}")
    print("You can play it with a command like:")
    print(f"aplay {output_path}")