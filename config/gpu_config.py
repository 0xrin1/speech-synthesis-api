"""
GPU-specific configuration overrides.
"""

from config.production import *

# GPU settings - use GPU 2 which has less usage
FORCE_GPU_DEVICE = 2
