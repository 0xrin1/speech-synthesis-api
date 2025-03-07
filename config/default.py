"""
Default configuration settings for the Speech API.
"""

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6000
RELOAD = False

# GPU settings
FORCE_GPU_DEVICE = 2  # Always use GPU device 2
MEMORY_RESERVE_PERCENTAGE = 0.9  # Reserve 90% of available GPU memory

# Model settings
PRIMARY_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # High-quality female voice
MULTI_SPEAKER_MODEL = "tts_models/en/vctk/vits"  # Multi-speaker with male voices
DEFAULT_SPEAKER = "p311"  # Very deep, clear male voice
FALLBACK_SPEAKER = "p326"  # Alternative deep male voice

# Audio settings
DEFAULT_SAMPLE_RATE = 22050
HIGH_QUALITY_SAMPLE_RATE = 48000
PAUSE_DURATION = 0.25  # Pause between sentences in seconds

# Path settings
OUTPUT_DIR = "output"