"""
Main server application for the Speech API.
"""

import logging
import os
import importlib.util
from fastapi import FastAPI
import uvicorn

from src.core import DeviceManager, ModelLoader, TTSEngine
from src.api import TTSRouter

# Configure logging
log_level = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app(config_module="config.production"):
    """
    Create and configure the FastAPI application.
    
    Args:
        config_module: Python module path to configuration
        
    Returns:
        FastAPI application instance
    """
    # Import configuration
    logger.info(f"Loading configuration from {config_module}")
    config = importlib.import_module(config_module)
    
    # Initialize FastAPI app
    app = FastAPI(title="Speech Synthesis API")
    
    # Configure components
    device_manager = DeviceManager(
        force_gpu_device=getattr(config, "FORCE_GPU_DEVICE", None),
        memory_reserve_percentage=getattr(config, "MEMORY_RESERVE_PERCENTAGE", 0.9)
    )
    
    # Set up device (GPU or CPU)
    device, memory_reservation = device_manager.setup()
    logger.info(f"Using device: {device}")
    
    # Store memory reservation in app state to prevent garbage collection
    if device.startswith("cuda"):
        app.state.reserved_memory = memory_reservation
    
    # Create model loader
    model_loader = ModelLoader(
        tts_model_name=getattr(config, "TTS_MODEL", "tts_models/en/vctk/vits"),
        default_speaker=getattr(config, "DEFAULT_SPEAKER", "p311"),
        fallback_speaker=getattr(config, "FALLBACK_SPEAKER", "p326"),
        device_manager=device_manager
    )
    
    # Load models
    success, error_message = model_loader.load_models()
    if not success:
        logger.error(f"Failed to load models: {error_message}")
    else:
        logger.info("Models loaded successfully")
    
    # Create TTS engine
    tts_engine = TTSEngine(
        model_loader=model_loader,
        device_manager=device_manager,
        default_sample_rate=getattr(config, "DEFAULT_SAMPLE_RATE", 22050),
        pause_duration=getattr(config, "PAUSE_DURATION", 0.25)
    )
    
    # Set up API routes
    tts_router = TTSRouter(
        tts_engine=tts_engine,
        default_speaker=getattr(config, "DEFAULT_SPEAKER", "p311")
    )
    
    # Include router in app
    app.include_router(tts_router.router)
    
    return app

def main():
    """Run the server application."""
    # Determine which config to use (default to production)
    config_module = os.environ.get("SPEECH_API_CONFIG", "config.production")
    
    # Create the app
    app = create_app(config_module)
    
    # Load configuration
    config = importlib.import_module(config_module)
    
    # Start the server
    logger.info(f"Starting server on {config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=config.RELOAD
    )

if __name__ == "__main__":
    main()