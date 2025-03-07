import torch
from TTS.api import TTS

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Initialize TTS
print("\nInitializing TTS...")
tts = TTS()

# List available models
print("\nAvailable TTS models:")
try:
    models = tts.list_models()
    if hasattr(models, "__iter__"):
        print(f"Found {len(list(models))} models")
        for model in models:
            print(f"- {model}")
    else:
        print("Cannot iterate through models, trying different approach...")
        
        # Try to list model types
        print("\nTTS attributes:")
        for attr in dir(tts):
            if not attr.startswith("_"):
                print(f"- {attr}")
                
        # List available TTS models by category
        print("\nAvailable model types:")
        print("- tts_models")
        print("- vocoder_models")
        print("- voice_conversion_models")
        print("- other_models")
        
        # Check for high-quality models
        print("\nChecking for high-quality models:")
        try:
            # Try XTTS (one of the best models)
            print("\nTrying XTTS models:")
            models_to_try = [
                "tts_models/multilingual/multi-dataset/xtts_v2",
                "tts_models/en/ljspeech/tacotron2-DDC_ph",
                "tts_models/en/ljspeech/glow-tts",
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/en/ljspeech/neural_hmm",
                "tts_models/en/vctk/vits",
                "tts_models/en/jenny/jenny"
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"\nTrying model: {model_name}")
                    model = TTS(model_name=model_name)
                    print(f"✅ Successfully loaded {model_name}")
                    
                    # Check if it has speakers
                    if hasattr(model, "speakers") and model.speakers:
                        print(f"Available speakers: {len(model.speakers)}")
                        male_speakers = [s for s in model.speakers if isinstance(s, str) and (s.startswith('p3') or 'male' in s.lower())]
                        if male_speakers:
                            print(f"Male speakers: {len(male_speakers)}")
                            print(f"Sample male speakers: {male_speakers[:5]}")
                    
                    # Clear model to save memory
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    
        except Exception as e:
            print(f"Error exploring models: {e}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nExploring available VITS models (high quality neural TTS):")
vits_models = [
    "tts_models/en/vctk/vits",  # Multi-speaker VITS (current model)
    "tts_models/en/ljspeech/vits",  # Single female speaker but higher quality
    "tts_models/en/vctk/vits--neon",  # Alternative VITS
]

for model_name in vits_models:
    try:
        print(f"\nTrying model: {model_name}")
        model = TTS(model_name=model_name)
        print(f"✅ Successfully loaded {model_name}")
        
        # Check if it has speakers
        if hasattr(model, "speakers") and model.speakers:
            print(f"Available speakers: {len(model.speakers)}")
            # Print first few speakers
            print(f"Sample speakers: {list(model.speakers)[:5]}")
            
        # Test generation with model
        text = "This is a high-quality text-to-speech sample using a large neural model on an RTX 3090 GPU."
        print(f"Generating sample speech with text: '{text}'")
        if hasattr(model, "speakers") and model.speakers:
            speaker = "p226" if "p226" in model.speakers else model.speakers[0]
            wav = model.tts(text=text, speaker=speaker)
            print(f"Generated audio with speaker {speaker}")
        else:
            wav = model.tts(text=text)
            print("Generated audio with default speaker")
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Clear model to save memory
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")

# Check model capabilities - specifically YourTTS which can do voice cloning
print("\nExploring YourTTS (voice cloning) model:")
try:
    yourtts_model = "tts_models/multilingual/multi-dataset/your_tts"
    model = TTS(model_name=yourtts_model)
    print(f"✅ Successfully loaded {yourtts_model}")
    
    if hasattr(model, "speakers") and model.speakers:
        print(f"Available speakers: {len(model.speakers)}")
        print(f"Sample speakers: {list(model.speakers)[:5]}")
    
    # Test capabilities
    print("YourTTS capabilities:")
    print("- Multi-lingual: Yes")
    print("- Voice cloning: Yes")
    print("- Emotion control: Limited")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ Failed to load YourTTS: {e}")

print("\nExploration complete!")