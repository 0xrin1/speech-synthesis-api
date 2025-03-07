import requests
import argparse
import os
import sys
import io

def speak_text(text, server_url="http://localhost:8000", output_file=None):
    """
    Send text to the speech API and save the resulting audio.
    
    Args:
        text (str): The text to convert to speech
        server_url (str): URL of the speech synthesis server
        output_file (str): Path to save the WAV file
        
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
    if len(sys.argv) < 2:
        print("Usage: python speech_client.py 'Your text here'")
        sys.exit(1)
        
    text = sys.argv[1]
    output_file = "/home/claudecode/speech-api/output/speech.wav"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Call API and save output
    speak_text(text, output_file=output_file)
    
    # Print path to the generated audio file
    print(f"\nAudio file generated: {output_file}")
    print("You can play it with a command like:")
    print(f"aplay {output_file}")