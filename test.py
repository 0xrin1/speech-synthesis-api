import requests
import sys
import os
import argparse

def speak(text, server_url, output_file, speaker_id=None, use_male_voice=True, 
        use_high_quality=True, max_gpu_memory=24, enhance_audio=True):
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
        params = {
            "text": text,
            "use_male_voice": use_male_voice,
            "use_high_quality": use_high_quality,
            "max_gpu_memory": max_gpu_memory,
            "enhance_audio": enhance_audio
        }
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
    parser.add_argument("--speaker", default="p311", 
                        help="Speaker ID for multi-speaker models (default: p311, deep male voice)")
    parser.add_argument("--use-male-voice", dest="use_male_voice", action="store_true",
                        help="Use male voice (default)")
    parser.add_argument("--use-female-voice", dest="use_male_voice", action="store_false",
                        help="Use female voice")
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True,
                        help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false",
                        help="Use lower quality for faster generation")
    parser.add_argument("--gpu-memory", dest="max_gpu_memory", type=int, default=24,
                        help="Maximum GPU memory to use in GB (1-24, default: 24)")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True,
                        help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false",
                        help="Skip additional audio enhancement")
    
    parser.set_defaults(use_male_voice=True, enhance_audio=True)
    
    args = parser.parse_args()
    
    speak(args.text, args.server, args.output, args.speaker, 
          use_male_voice=args.use_male_voice, 
          use_high_quality=args.use_high_quality,
          max_gpu_memory=args.max_gpu_memory,
          enhance_audio=args.enhance_audio)