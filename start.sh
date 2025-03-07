#!/bin/bash

# Activate conda environment
source /home/claudecode/miniconda3/etc/profile.d/conda.sh
conda activate speech-api

# Make sure we have pydub for the client
pip install pydub

# Start the API server
cd /home/claudecode/speech-api/app
python main.py