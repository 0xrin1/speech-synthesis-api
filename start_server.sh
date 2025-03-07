#!/bin/bash

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

# Set config to production
export SPEECH_API_CONFIG="config.production"

# Start the server
python -m src.server