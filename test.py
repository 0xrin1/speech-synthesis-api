import requests
import sys
import os
import argparse

def speak(text, server_url, output_file):
    """Call the TTS API and generate audio"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Call the API
        print(f"Sending request to {server_url}/tts")
        response = requests.get(
            f"{server_url}/tts",
            params={"text": text},
            stream=True
        )
        
        # Check if successful
        if response.status_code == 200:
            # Save to file
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Audio generated successfully and saved to {output_file}")
            print(f"You can play it with a media player or command line tools")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Client")
    parser.add_argument("text", nargs="?", default="Hello, I am Claude, your AI assistant. How can I help you today?", 
                        help="Text to convert to speech")
    parser.add_argument("--server", "-s", default="http://localhost:8080", 
                        help="Server URL (default: http://localhost:8080)")
    parser.add_argument("--output", "-o", default="./output/speech.wav", 
                        help="Output file path (default: ./output/speech.wav)")
    
    args = parser.parse_args()
    
    speak(args.text, args.server, args.output)