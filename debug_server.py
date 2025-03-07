import os
import io
import subprocess
import torch
from TTS.api import TTS

# Configure for high quality output with a deep male voice
# p326 is a particularly good male voice based on testing
default_speaker = "p326"  # Deep male voice

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize TTS model with highest quality settings
try:
    # Using Coqui TTS with the VITS model - highest quality
    # VITS is one of the best neural TTS models, perfect for utilizing GPU acceleration
    tts = TTS(model_name="tts_models/en/vctk/vits")
    
    # Set device manually if possible - use entire GPU for maximum quality
    if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'to'):
        # Pin memory for faster inference
        torch.set_grad_enabled(False)  # Disable gradients for inference
        torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
        
        # Move model to GPU with optimizations
        tts.synthesizer.to(device)
        
        # Set batch size to 1 for highest quality per sample
        if hasattr(tts.synthesizer, 'batch_size'):
            tts.synthesizer.batch_size = 1
    
    print(f"High-quality TTS model loaded successfully on {device}!")
    print(f"Using default male voice: {default_speaker}")
    
    print("Available speakers:", tts.speakers)
    
    # Test synthesis with a sample text
    print("Testing speech synthesis...")
    with torch.cuda.amp.autocast(enabled=True):
        wav = tts.tts(text="This is a test of the high-quality speech synthesis", speaker=default_speaker)
    
    print("Success! Speech generated.")
    
    # Save test output
    output_path = "output/test_speech.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts.synthesizer.save_wav(wav, output_path)
    print(f"Saved test output to {output_path}")
    
except Exception as e:
    print(f"Error loading TTS model: {e}")