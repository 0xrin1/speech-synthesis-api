from TTS.api import TTS

print("Available models:")
try:
    models = TTS().list_models()
    for model in models:
        print(f"- {model}")
except:
    print("Could not list models")

print("\nChecking for XTTS v2:")
try:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("XTTS v2 available!")
except Exception as e:
    print(f"XTTS v2 not available: {e}")

print("\nChecking for high-quality VITS model:")
try:
    tts = TTS("tts_models/en/vctk/vits")
    print("VITS model available!")
    print("Available speakers:", tts.speakers)
    print("\nMale speakers (p3xx):", [s for s in tts.speakers if s.startswith('p3')])
except Exception as e:
    print(f"VITS not available: {e}")