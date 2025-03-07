import requests
import sys
import os
import argparse

def speak(text, server_url, output_file, speaker_id=None):
    """Call the TTS API and generate audio"""
    try:
        # Process output path - convert relative path to absolute if needed
        abs_output_file = output_file
        if not os.path.isabs(abs_output_file):
            # Make relative to current script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_output_file = os.path.normpath(os.path.join(script_dir, abs_output_file))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(abs_output_file), exist_ok=True)
        
        # Set up parameters
        params = {"text": text}
        if speaker_id:
            params["speaker"] = speaker_id
        
        # Call the API
        print(f"Sending request to {server_url}/tts with params {params}")
        response = requests.get(
            f"{server_url}/tts",
            params=params,
            stream=True
        )
        
        # Check if successful
        if response.status_code == 200:
            # Save to file
            with open(abs_output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Audio generated successfully and saved to {abs_output_file}")
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
    parser.add_argument("--server", "-s", default="http://localhost:6000", 
                        help="Server URL (default: http://localhost:6000)")
    parser.add_argument("--output", "-o", default="./output/speech.wav", 
                        help="Output file path (default: ./output/speech.wav)")
    parser.add_argument("--speaker", default="p335", 
                        help="Speaker ID for multi-speaker models (default: p335)")
    
    args = parser.parse_args()
    
    speak(args.text, args.server, args.output, args.speaker)