"""
Device manager module for handling GPU/CPU selection and memory management.
"""

import os
import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Manages computation devices (GPU/CPU) and memory allocation.
    """
    
    def __init__(self, force_gpu_device: int = None, memory_reserve_percentage: float = 0.9):
        """
        Initialize the device manager.
        
        Args:
            force_gpu_device: Force a specific GPU device (if None, will select automatically)
            memory_reserve_percentage: Percentage of GPU memory to reserve (0.0-1.0)
        """
        self.force_gpu_device = force_gpu_device
        self.memory_reserve_percentage = memory_reserve_percentage
        self.device = None
        self.memory_reservation = None
        
    def setup(self) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Set up and configure the computation device (GPU or CPU).
        
        Returns:
            Tuple containing (device_string, memory_reservation_tensor)
        """
        try:
            if torch.cuda.is_available():
                # Set up GPU
                return self._setup_gpu()
            else:
                # Fall back to CPU
                logger.warning("CUDA not available, using CPU instead.")
                logger.warning("Speech synthesis will be much slower without GPU acceleration.")
                return "cpu", None
        except Exception as e:
            # Handle permission issues or other CUDA errors
            logger.error(f"Error accessing CUDA: {e}")
            logger.warning("Falling back to CPU mode (speech synthesis will be much slower).")
            return "cpu", None
    
    def _setup_gpu(self) -> Tuple[str, torch.Tensor]:
        """
        Set up GPU device and reserve memory.
        
        Returns:
            Tuple containing (device_string, memory_reservation_tensor)
        """
        # Use the forced GPU device if specified
        if self.force_gpu_device is not None:
            device_id = self.force_gpu_device
            device_str = f"cuda:{device_id}"
            
            # Configure CUDA environment
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            logger.info(f"Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
            
            # Verify we're actually using it
            torch.cuda.set_device(device_id)
            logger.info(f"Active GPU: {torch.cuda.current_device()}")
            
            # Reserve memory
            with torch.no_grad():
                # Get total memory on the device
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                # Reserve memory
                reserve_size = int(self.memory_reserve_percentage * total_memory)
                # Create a tensor to hold the reservation
                dummy = torch.empty(reserve_size, dtype=torch.uint8, device=device_str)
                logger.info(f"Reserved {reserve_size/1024**3:.1f}GB GPU memory")
                
                self.device = device_str
                self.memory_reservation = dummy
                return device_str, dummy
        else:
            # Auto-select GPU with most available memory
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.warning("No CUDA devices available, using CPU instead.")
                return "cpu", None
                
            # Find GPU with most free memory
            max_free_memory = 0
            selected_device = 0
            
            for i in range(device_count):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                free_memory = torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    selected_device = i
            
            device_str = f"cuda:{selected_device}"
            logger.info(f"Auto-selected CUDA device {selected_device}: {torch.cuda.get_device_name(selected_device)}")
            
            # Reserve memory on selected device
            with torch.no_grad():
                total_memory = torch.cuda.get_device_properties(selected_device).total_memory
                reserve_size = int(self.memory_reserve_percentage * total_memory)
                dummy = torch.empty(reserve_size, dtype=torch.uint8, device=device_str)
                logger.info(f"Reserved {reserve_size/1024**3:.1f}GB GPU memory")
                
                self.device = device_str
                self.memory_reservation = dummy
                return device_str, dummy
    
    def optimize_for_inference(self):
        """
        Configure PyTorch for optimal inference performance.
        """
        # Optimize GPU settings for inference
        torch.set_grad_enabled(False)  # Disable gradients for inference
        torch.backends.cudnn.benchmark = True  # Use cuDNN auto-tuner
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster math
        
    def get_memory_info(self):
        """
        Get memory usage information for the current device.
        
        Returns:
            Dict containing memory information
        """
        if not torch.cuda.is_available() or self.device is None or self.device == "cpu":
            return {"device": "cpu", "memory_info": "N/A"}
            
        device_id = torch.cuda.current_device()
        return {
            "device": self.device,
            "name": torch.cuda.get_device_name(device_id),
            "index": device_id,
            "memory_allocated": f"{torch.cuda.memory_allocated(device_id) / 1024**3:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved(device_id) / 1024**3:.2f} GB",
            "memory_total": f"{torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB"
        }
    
    def empty_cache(self):
        """
        Clear GPU cache to free up memory.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __str__(self):
        """String representation of the device manager."""
        if self.device is None:
            return "DeviceManager (not initialized)"
        return f"DeviceManager (device={self.device})"