# Speech Synthesis API

High-quality neural text-to-speech API with GPU acceleration and deep male voices.

## Features

- High-quality speech synthesis using neural TTS models
- Multiple voice options (female voice and various male voices)
- GPU acceleration for fast processing
- Audio enhancement for improved quality
- Simple REST API with both GET and POST endpoints
- Command-line clients for easy testing and interaction
- Modular, maintainable architecture

## Quick Setup

1. Make sure you have conda installed
2. Run the setup script to create and activate the environment:

```bash
# Install dependencies
./setup_env.sh

# Start the server
./start_server.sh
```

## Server Setup Options

```bash
# Start in production mode with GPU acceleration (port 6000)
./start_server.sh

# Start in development mode (port 8080, auto-reload enabled)
./start_server.sh --dev

# Start with debug logging enabled
./start_server.sh --debug

# Combine options
./start_server.sh --dev --debug
```

**⚠️ IMPORTANT:** This server requires an NVIDIA GPU with at least 8GB VRAM. The server will fail to start if no GPU is detected. GPU acceleration requires sudo access.

The server will automatically prompt for your password to gain the necessary permissions for GPU access.

## API Endpoints

### GET /tts

Simple endpoint that accepts text as a query parameter.

Example:
```
GET http://localhost:6000/tts?text=Hello%20world&speaker=p311&use_male_voice=true
```

Parameters:
- `text`: Text to convert to speech (required)
- `speaker`: Speaker ID for multi-speaker models (default: p311)
- `use_male_voice`: Use male voice (true) or female voice (false)
- `use_high_quality`: Use highest quality settings (default: true)
- `enhance_audio`: Apply additional audio enhancement (default: true)

### POST /tts

Advanced endpoint that accepts JSON with text and optional parameters.

Example:
```json
POST http://localhost:6000/tts
Content-Type: application/json

{
  "text": "Hello world",
  "voice_id": "p311",
  "speed": 1.0,
  "use_high_quality": true,
  "use_male_voice": true,
  "enhance_audio": true
}
```

## Client Usage

### Test Client

```bash
# Basic usage with default text
python test.py

# Custom text
python test.py "Hello, this is a custom message"

# Connect to a remote server
python test.py "Hello world" --server http://server-ip:6000

# Custom output path
python test.py "Hello world" --output ~/Desktop/speech.wav

# Advanced options
python test.py "Hello world" --advanced --speed 0.8 --female
```

### CLI Client

```bash
# Basic usage
python -m src.client.cli_client "Hello world"

# Advanced usage
python -m src.client.cli_client "Hello world" --voice p364 --speed 1.2 --female
```

### Playback Client (with automatic audio playback)

```bash
# Play audio immediately
python -m src.client.playback_client "Hello world"

# Save to file without playing
python -m src.client.playback_client "Hello world" --no-play --output ~/Desktop/speech.wav
```

## Project Structure

```
/speech-api/
├── config/               # Configuration files
├── src/                  # Source code
│   ├── api/              # API endpoints and models
│   ├── client/           # Client implementations
│   ├── core/             # Core TTS functionality
│   └── server.py         # Main server application
├── scripts/              # Helper scripts
├── tests/                # Test suite
├── output/               # Default output directory
├── start_server.sh       # Server startup script
└── test.py               # Simple test client
```

## Configuration

Configuration is stored in the `config` directory:

- `default.py`: Base configuration
- `development.py`: Development settings
- `production.py`: Production settings

Set the environment variable `SPEECH_API_CONFIG` to choose a configuration:

```bash
export SPEECH_API_CONFIG="config.development"
```

## Deployment

For production deployment:

1. Clone this repository on your server
2. Run `./setup_env.sh` to install dependencies
3. Start the server with `./start_server.sh`
4. Make sure port 6000 is open in your firewall if accessing remotely
5. Use the client with `python test.py --server http://your-server-ip:6000`

## System Requirements

- Python 3.10+
- Conda package manager
- NVIDIA GPU with CUDA support (minimum 8GB VRAM, 24GB+ recommended for best performance)
- Approximately 4GB of RAM for model loading
- Sudo access for GPU utilization