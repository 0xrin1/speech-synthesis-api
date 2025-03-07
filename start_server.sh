#!/bin/bash

# Default configuration
MODE="production"
DEBUG=false
USE_GPU=true

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    --no-gpu)
      USE_GPU=false
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--dev] [--debug] [--no-gpu]"
      echo "  --dev     Start in development mode (port 8080, auto-reload)"
      echo "  --debug   Enable debug output"
      echo "  --no-gpu  Don't use GPU acceleration, even if available"
      exit 1
      ;;
  esac
done

# Function to start server as the current user (no GPU)
start_server_cpu() {
    # Activate Conda environment if available
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
        conda activate speech-api || echo "Environment 'speech-api' not found, skipping activation"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
        conda activate speech-api || echo "Environment 'speech-api' not found, skipping activation"
    fi

    # Start the server
    python -m src.server
}

# Function to start server with GPU access (requires sudo)
start_server_gpu() {
    # Create the GPU config file
    cat > /home/claudecode/speech-api/config/gpu_config.py << EOF
"""
GPU-specific configuration overrides.
"""

from config.${MODE} import *

# Use GPU 2 which has less usage
FORCE_GPU_DEVICE = 2
EOF
    
    # Start with sudo to get GPU access
    echo -e "${BLUE}Requesting sudo access to use GPU...${NC}"
    sudo bash -c "
        # Set up environment
        source /home/claudecode/miniconda3/etc/profile.d/conda.sh
        conda activate speech-api
        
        # Set environment variables
        export PYTHONPATH='/home/claudecode/speech-api'
        export SPEECH_API_CONFIG='config.gpu_config'
        export CUDA_VISIBLE_DEVICES='2'
        export LOGLEVEL='${LOGLEVEL}'
        
        # Start the server
        cd /home/claudecode/speech-api
        python -m src.server
    "
}

# Create output directory
mkdir -p output
chmod 777 output

# Set configuration based on mode
export SPEECH_API_CONFIG="config.${MODE}"

# Print startup information
echo -e "${BLUE}Starting Speech API server in ${MODE} mode${NC}"
if [ "$MODE" = "development" ]; then
    echo -e "${BLUE}Server will run on port 8080 with auto-reload enabled${NC}"
else
    echo -e "${BLUE}Server will run on port 6000${NC}"
fi

# Enable debug logging if requested
if [ "$DEBUG" = true ]; then
    export LOGLEVEL="DEBUG"
    echo -e "${BLUE}Debug logging enabled${NC}"
else
    export LOGLEVEL="INFO"
fi

# Check if we should try to use GPU
if [ "$USE_GPU" = true ]; then
    # Check if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}NVIDIA GPU detected, starting server with GPU acceleration${NC}"
        start_server_gpu
    else
        echo -e "${RED}No NVIDIA GPU detected, falling back to CPU mode${NC}"
        start_server_cpu
    fi
else
    echo -e "${BLUE}Using CPU mode as requested${NC}"
    start_server_cpu
fi