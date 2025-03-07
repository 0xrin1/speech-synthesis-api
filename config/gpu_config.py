"""
GPU-specific configuration overrides.
"""

from config.production import *

# Use GPU 0 since CUDA_VISIBLE_DEVICES remaps devices
FORCE_GPU_DEVICE = 0
