# Speech Synthesis API

A GPU-accelerated API server for text-to-speech synthesis using state-of-the-art models from Coqui TTS.

## Features

- Convert text to speech using GPU acceleration (if available)
- Multiple voices support (depending on model)
- Simple REST API with both GET and POST endpoints
- Streaming audio output
- Command-line client for testing and demonstration

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
# Start with custom host and port
./start_server.sh --host 127.0.0.1 --port 9000

# Enable auto-reload for development
./start_server.sh --reload
```

## API Endpoints

### GET /tts

Simple endpoint that accepts text as a query parameter.

Example:
```
GET http://localhost:8080/tts?text=Hello%20world
```

### POST /tts

Advanced endpoint that accepts JSON with text and optional parameters.

Example:
```json
POST http://localhost:8080/tts
{
  "text": "Hello world",
  "voice_id": "p336",  # If supported by model
  "speed": 1.0
}
```

## Client Usage

The included test.py client can be used to test the API:

```bash
# Basic usage with default text
python test.py

# Custom text
python test.py "Hello, this is a custom message"

# Connect to a remote server
python test.py "Hello world" --server http://server-ip:8080

# Custom output path
python test.py "Hello world" --output ~/Desktop/speech.wav
```

## Deployment

For production deployment:

1. Clone this repository on your server
2. Run `./setup_env.sh` to install dependencies
3. Start the server with `./start_server.sh`
4. Make sure port 8080 is open in your firewall if accessing remotely
5. Use the client with `python test.py --server http://your-server-ip:8080`

## System Requirements

- Python 3.10+
- Conda package manager
- CUDA-capable GPU recommended for faster processing (but not required)
- Approximately 2GB of RAM for model loading