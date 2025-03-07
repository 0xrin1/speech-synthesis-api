#!/bin/bash

# Activate conda environment
source /home/claudecode/miniconda3/etc/profile.d/conda.sh
conda activate speech-api

# Create a test output directory if it doesn't exist
mkdir -p /home/claudecode/speech-api/output

# Run the client with a sample text
cd /home/claudecode/speech-api/app
python client.py "Hello, this is a test of the speech synthesis API!" --output /home/claudecode/speech-api/output/test_speech.wav