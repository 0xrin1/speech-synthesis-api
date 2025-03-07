import os
import torch
from TTS.api import TTS

# Create output directory
os.makedirs("output", exist_ok=True)

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Try to load YourTTS, which is a high-quality model with voice cloning abilities
print("Loading YourTTS model...")
try:
    model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to("cuda")
    print("✅ YourTTS model loaded successfully!")
    print(f"Memory usage after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # List available speakers
    if hasattr(model, "speakers"):
        print(f"Available speakers: {len(model.speakers)}")
        print(f"First few speakers: {list(model.speakers)[:5]}")
    
    # Generate samples for a male voice
    try:
        text = "This is a test of the YourTTS model, which is a high-quality speech synthesis system using a deep neural network. It can generate much more natural sounding speech than simpler models."
        print(f"Generating speech with YourTTS: '{text}'")
        
        # Find a male speaker (if available)
        male_speakers = ["p330", "VCTK_p270", "7021_88884", "7021_85628", "7021"]
        speaker = None
        for s in male_speakers:
            if s in model.speakers:
                speaker = s
                break
        
        if not speaker and hasattr(model, "speakers") and model.speakers:
            speaker = list(model.speakers)[0]
            
        if speaker:
            print(f"Using speaker: {speaker}")
            wav = model.tts(text=text, speaker=speaker)
        else:
            print("No speaker found, using default speaker")
            wav = model.tts(text=text)
            
        # Save output
        output_file = "output/your_tts_sample.wav"
        model.synthesizer.save_wav(wav, output_file)
        print(f"✅ Audio saved to {output_file}")
        print(f"Final memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error generating YourTTS speech: {e}")
except Exception as e:
    print(f"Error loading YourTTS: {e}")
    print("Trying VITS model instead...")
    
    # Fall back to VITS model and try different speakers
    try:
        model = TTS(model_name="tts_models/en/vctk/vits").to("cuda")
        print("✅ VITS model loaded successfully!")
        print(f"Memory usage after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Test different male voices to find the best deep male voice
        text = "This is a test of the VITS model, which is a high-quality speech synthesis system using a deep neural network. It can generate natural sounding speech with this deep male voice."
        male_speakers = ["p226", "p270", "p271", "p232", "p326", "p302", "p311", "p364"]
        
        for speaker in male_speakers:
            try:
                if speaker in model.speakers:
                    print(f"Testing speaker: {speaker}")
                    output_file = f"output/vits_male_{speaker}.wav"
                    wav = model.tts(text=text, speaker=speaker)
                    model.synthesizer.save_wav(wav, output_file)
                    print(f"✅ Audio saved to {output_file}")
            except Exception as e:
                print(f"Error with speaker {speaker}: {e}")
    except Exception as e:
        print(f"Error loading VITS: {e}")

print("Test completed!")