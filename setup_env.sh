#!/bin/bash

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
conda create -y -n speech-api python=3.10

# Activate environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh. Please activate the environment manually."
    exit 1
fi

conda activate speech-api

# Install required packages
pip install torch torchaudio fastapi uvicorn TTS requests

echo "Environment setup complete!"
echo "To start the server, run: ./start_server.sh"
echo "To test the client, run: python test.py 'Your text here' --server http://server-ip:8080"