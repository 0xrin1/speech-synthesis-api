#!/usr/bin/env python3
"""
Simple test client for the Speech API.
"""

import os
import sys
import argparse
import logging

# Add src to Python path for imports from the new structure
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.client import CLIClient

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Text-to-Speech Test Client")
    parser.add_argument("text", nargs="?", default="Hello, I am Claude, your AI assistant. How can I help you today?", 
                        help="Text to convert to speech")
    parser.add_argument("--server", "-s", default="http://localhost:6000", 
                        help="Server URL (default: http://localhost:6000)")
    parser.add_argument("--output", "-o", default="./output/speech.wav", 
                        help="Output file path (default: ./output/speech.wav)")
    parser.add_argument("--speaker", default="p311", 
                        help="Speaker ID for multi-speaker models (default: p311, deep male voice)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed factor (1.0 is normal, only used with --advanced)")
    parser.add_argument("--advanced", action="store_true",
                        help="Use advanced POST endpoint with more options")
    
    # Voice and quality options
    parser.add_argument("--male", dest="use_male_voice", action="store_true", default=True,
                        help="Use male voice (default)")
    parser.add_argument("--female", dest="use_male_voice", action="store_false",
                        help="Use female voice")
    parser.add_argument("--high-quality", dest="use_high_quality", action="store_true", default=True,
                        help="Use highest quality settings (default)")
    parser.add_argument("--low-quality", dest="use_high_quality", action="store_false",
                        help="Use lower quality for faster generation")
    parser.add_argument("--enhance", dest="enhance_audio", action="store_true", default=True,
                        help="Apply additional audio enhancement (default)")
    parser.add_argument("--no-enhance", dest="enhance_audio", action="store_false",
                        help="Skip additional audio enhancement")
    
    args = parser.parse_args()
    
    # Create client
    client = CLIClient(server_url=args.server)
    
    # Print request info
    print(f"Sending request to {args.server}/tts")
    print(f"Text: {args.text}")
    print(f"Voice: {'Male' if args.use_male_voice else 'Female'}")
    print(f"Speaker: {args.speaker}")
    print(f"Quality: {'High' if args.use_high_quality else 'Low'}")
    print(f"Enhancement: {'Enabled' if args.enhance_audio else 'Disabled'}")
    
    # Use advanced options if specified
    if args.advanced or args.speed != 1.0:
        print(f"Using advanced options (POST endpoint)")
        print(f"Speed: {args.speed}")
        
        client.advanced_speak_text(
            text=args.text,
            output_file=args.output,
            voice_id=args.speaker,
            speed=args.speed,
            use_male_voice=args.use_male_voice,
            use_high_quality=args.use_high_quality,
            enhance_audio=args.enhance_audio
        )
    else:
        client.speak_text(
            text=args.text,
            output_file=args.output,
            speaker=args.speaker,
            use_male_voice=args.use_male_voice,
            use_high_quality=args.use_high_quality,
            enhance_audio=args.enhance_audio
        )
    
    # Print output info
    abs_path = os.path.abspath(args.output)
    print(f"\nAudio saved to: {abs_path}")
    print(f"You can play it with a media player or command line tools")