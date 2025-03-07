#!/bin/bash

# Default configuration
MODE="production"
DEBUG=false

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
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Usage: $0 [--dev] [--debug]"
      echo "  --dev     Start in development mode (port 8080, auto-reload)"
      echo "  --debug   Enable debug output"
      exit 1
      ;;
  esac
done

# This server requires GPU acceleration - removed CPU fallback

# Function to start server with GPU access (requires sudo)
start_server_gpu() {
    # Start with sudo to get GPU access
    echo -e "${BLUE}Requesting sudo access to use GPU...${NC}"
    
    sudo bash -c "
        # Set up environment
        source /home/claudecode/miniconda3/etc/profile.d/conda.sh
        conda activate speech-api
        
        # Create GPU config directly
        mkdir -p /home/claudecode/speech-api/config
        cat > /home/claudecode/speech-api/config/gpu_config.py << EOF
\"\"\"
GPU-specific configuration overrides.
\"\"\"

from config.${MODE} import *

# Use GPU 0 since CUDA_VISIBLE_DEVICES remaps devices
FORCE_GPU_DEVICE = 0
EOF
        
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

# Check if port is already in use
PORT=6000
if [ "$MODE" = "development" ]; then
    PORT=8080
fi

# Check and kill any process using our port
if lsof -i:${PORT} > /dev/null; then
    echo -e "${BLUE}Port ${PORT} is already in use. Stopping existing process...${NC}"
    # Find PID of process using our port
    PID=$(lsof -t -i:${PORT})
    if [ ! -z "$PID" ]; then
        echo -e "${BLUE}Stopping process ${PID}...${NC}"
        kill ${PID}
        # Wait for the port to be freed
        sleep 3
    fi
fi

# Verify GPU is available, exit if not
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: No NVIDIA GPU detected. This server requires GPU acceleration.${NC}"
    echo -e "${RED}Please make sure you have an NVIDIA GPU and the proper drivers installed.${NC}"
    exit 1
fi

# Check GPU access permissions
if ! sudo -n nvidia-smi &> /dev/null; then
    echo -e "${RED}GPU detected but requires sudo permissions.${NC}"
fi

echo -e "${GREEN}NVIDIA GPU detected, starting server with GPU acceleration${NC}"
start_server_gpu