import os
import torch
from TTS.api import TTS
import time

# Create output directory
os.makedirs("output", exist_ok=True)

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Models to try
models_to_test = [
    "tts_models/en/ljspeech/tacotron2-DDC",  # High-quality Tacotron2 model with Dynamic Convolution Decoder
    "tts_models/en/ljspeech/glow-tts",       # GlowTTS model - high quality with faster inference
    "tts_models/en/ljspeech/fast_pitch",     # FastPitch model - very fast with good quality
    "tts_models/en/vctk/fast_pitch",         # FastPitch with multiple speakers
]

# Test text
text = "This is a test of a larger, more advanced neural text-to-speech model. It should sound much more natural and use more of the GPU's computational power for higher quality speech synthesis."

# Test each model
for model_name in models_to_test:
    print(f"\n\n======== Testing model: {model_name} ========")
    try:
        # Load model
        print(f"Loading model...")
        start_time = time.time()
        model = TTS(model_name=model_name).to("cuda")
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print(f"Memory usage after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Check if it's a multi-speaker model
        has_speakers = hasattr(model, "speakers") and model.speakers
        if has_speakers:
            print(f"This is a multi-speaker model with {len(model.speakers)} speakers")
            speaker_param = {"speaker": model.speakers[0]} if model.speakers else {}
        else:
            print("This is a single speaker model")
            speaker_param = {}
        
        # Generate speech
        print(f"Generating speech...")
        start_time = time.time()
        
        # Optimize with torch.amp for better performance and quality
        with torch.cuda.amp.autocast():
            if has_speakers:
                wav = model.tts(text=text, **speaker_param)
            else:
                wav = model.tts(text=text)
                
        gen_time = time.time() - start_time
        print(f"✅ Speech generated in {gen_time:.2f} seconds")
        
        # Save output
        output_file = f"output/{model_name.replace('/', '_')}.wav"
        model.synthesizer.save_wav(wav, output_file)
        print(f"✅ Audio saved to {output_file}")
        
        # Memory usage
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
        print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        
print("\n\nNow testing Tacotron2-DDC with the Griffin-Lim and MultiBand-MelGAN vocoders")
try:
    # Two-step synthesis with high-quality vocoder
    print("\n======== Testing Tacotron2-DDC with High-Quality Vocoder ========")
    
    # Load TTS model
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    tts.to("cuda")
    print(f"TTS model loaded. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate mel spectrogram
    print("Generating mel spectrogram...")
    mel = tts.synthesizer.tts(text)
    print(f"Mel spectrogram generated. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate audio with Griffin-Lim (baseline)
    print("Generating audio with Griffin-Lim vocoder...")
    wav = tts.synthesizer.vocoder_model.spec_to_wav(mel)
    tts.synthesizer.save_wav(wav, "output/tacotron2_griffin_lim.wav")
    print(f"Griffin-Lim audio saved. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Load MultiBand-MelGAN vocoder
    tts.vocoder_model = TTS("vocoder_models/en/ljspeech/multiband-melgan")
    tts.vocoder_model.to("cuda")
    print(f"MultiBand-MelGAN vocoder loaded. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Generate audio with MultiBand-MelGAN
    print("Generating audio with MultiBand-MelGAN vocoder...")
    wav = tts.vocoder_model.inference(mel)
    tts.synthesizer.save_wav(wav, "output/tacotron2_multiband_melgan.wav")
    print(f"MultiBand-MelGAN audio saved. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Load HiFi-GAN vocoder if available
    try:
        print("Trying to load HiFi-GAN vocoder...")
        tts.vocoder_model = TTS("vocoder_models/en/ljspeech/hifigan_v1")
        tts.vocoder_model.to("cuda")
        print(f"HiFi-GAN vocoder loaded. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Generate audio with HiFi-GAN
        print("Generating audio with HiFi-GAN vocoder...")
        wav = tts.vocoder_model.inference(mel)
        tts.synthesizer.save_wav(wav, "output/tacotron2_hifigan.wav")
        print(f"HiFi-GAN audio saved. Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"Error loading HiFi-GAN vocoder: {e}")
    
    # Clear memory
    del tts
    torch.cuda.empty_cache()

except Exception as e:
    print(f"Error testing Tacotron2 with vocoders: {e}")

print("\nTesting completed!")