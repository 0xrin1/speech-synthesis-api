#!/bin/bash

# Default configuration
MODE="production"
DEBUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      MODE="development"
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dev] [--debug]"
      echo "  --dev    Start in development mode (port 8080, auto-reload)"
      echo "  --debug  Enable debug output"
      exit 1
      ;;
  esac
done

# Activate Conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate speech-api || echo "Environment 'speech-api' not found, skipping activation"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate speech-api || echo "Environment 'speech-api' not found, skipping activation"
fi

# Create output directory
mkdir -p output

# Set configuration based on mode
export SPEECH_API_CONFIG="config.${MODE}"

# Print startup information
echo "Starting Speech API server in ${MODE} mode"
if [ "$MODE" = "development" ]; then
    echo "Server will run on port 8080 with auto-reload enabled"
else
    echo "Server will run on port 6000"
fi

# Enable debug logging if requested
if [ "$DEBUG" = true ]; then
    export LOGLEVEL="DEBUG"
    echo "Debug logging enabled"
else
    export LOGLEVEL="INFO"
fi

# Start the server
python -m src.server