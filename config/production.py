"""
Production configuration settings for the Speech API.
"""

from config.default import *

# Disable auto-reload for production
RELOAD = False

# Use the standard port for production
SERVER_PORT = 6000

# Disable debugging
DEBUG = False