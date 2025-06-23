# Builtins
import threading
import os
import gc
# Installed
import torch
import numpy as np
# Local
from src.config import CONFIG
from src.core.logger import CORE_LOGGER
# Types
from typing import Literal


"""

Explanation of Changes
Locking Mechanism:

A class-level lock (_lock) is created using threading.Lock(). This lock will be used to synchronize access to the singleton instance creation.
Double-Checked Locking:

The __new__ method first checks if _instance is None. If it is, it acquires the lock and checks again (this is the double-checked locking pattern). This ensures that only one thread can create the instance at a time.
Initialization Check:

The __init__ method still checks if the instance has been initialized to prevent overwriting the param1 value.

"""

"""

Usage
With this implementation, you can safely create instances of SingletonTryouts from multiple threads without risking the creation of multiple instances:

def create_singleton(param):
    singleton = SingletonTryouts(param)

# Example usage in multiple threads
thread1 = threading.Thread(target=create_singleton, args=("first",))
thread2 = threading.Thread(target=create_singleton, args=("second",))

thread1.start()
thread2.start()

thread1.join()
thread2.join()


"""

"""

This thread-safe implementation ensures that even if multiple threads attempt to create an instance of SingletonTryouts simultaneously, only one instance will be created, and the initialization will occur only once. This is a common pattern used in singleton implementations to ensure both correctness and safety in multi-threaded environments.

"""

"""

Con este codigo testeo que no se creen mas clases:

    singleton = UNet(model_name="Ayax")
    new_singleton = UNet(model_name="Prok")

    print(singleton is new_singleton)
    print(singleton.__hash__())
    print(new_singleton.__hash__())

    singleton.singl_variable = "Singleton Variable"
    print(new_singleton.singl_variable)

    print("---------------")

    print(f"{singleton.__hash__()} -- {singleton.model_name}")
    print(f"{new_singleton.__hash__()} -- {new_singleton.model_name}")

"""



class VisionModel(object):

    _instance = None
    _lock = threading.Lock() # Lock for thread safety
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Acquire the lock to ensure thread safety
            with cls._lock:
                # Double check
                if cls._instance is None:
                    cls._instance = super(VisionModel, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str=None,
        model_type: Literal['unet', 'vit']='unet',
        device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        if not self._initialized:
            CORE_LOGGER.info("Initializing VisionModel class...")
            if model_type is None:
                CORE_LOGGER.error("model_type not supported. Available options: ['unet', 'vit']")
                raise ValueError("model_type not supported. Available options: ['unet', 'vit']")
            self.model_name = model_type if model_name is None else model_name
            self.model_type = model_type
            self.path = os.path.join(CONFIG.VISION_NN_FOLDER, self.model_type, f'{self.model_name}.pt')
            self.device = device
            self.model = None
            try:
                self.load()
            except Exception as e:
                CORE_LOGGER.error(f"There was a problem loading the vision model: {e}")
                raise Exception(f"There was a problem loading the vision model: {e}")
            
            CORE_LOGGER.info(self.__str__())
            self._initialized = True

    def __str__(self):
        return f"VisionModel class initialized with params (model_name={self.model_name}, model_type={self.model_type}, device={self.device}, path={self.path})"
    
    # To load the complete model later:
    def load(self):
        """Load a complete model including architecture and weights"""
        if self.is_loaded():
            CORE_LOGGER.warning(f"{self.path} is already loaded, no further actions will be executed here")
            # raise RuntimeError(f"{self.path} is already loaded, no further actions will be executed here")
        CORE_LOGGER.info(f"DEVICe: {self.device}")
        try:
            self.model = torch.load(self.path, weights_only=False, map_location=self.device)
        except:
            self.model = torch.load(self.path.replace('pt', 'pth'), weights_only=False, map_location=self.device)
        self.model.eval()  # Set to evaluation mode
        self.model = self.model.to(self.device)
        CORE_LOGGER.info(f"{self.path} loaded into {self.device} device")

    def unload(self):
        """Unload the model from GPU and free up RAM"""
        self.model = None
        CORE_LOGGER.info(f"{self.model_name} unloaded. Cleaning remaining data...")
        torch.cuda.empty_cache() # Clear the GPU memory cache
        gc.collect() # Collect garbage

    def is_loaded(self):
        """Checks if the model is loaded on GPU"""
        return True if self.model else False
    
    def make_prediction(self, image):
        if not self.is_loaded:
            self.load()
        
        CORE_LOGGER.info(f"Using {self.model_name} to make predictions...")
        with torch.no_grad():  # No gradients needed for evaluation
            # Convert the NumPy array to a PyTorch tensor
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)

            # Convert to float32 if not already
            if image.dtype != torch.float32:
                image = image.float()

            # Reshape the image tensor to (1, 1, 8, 128, 128)
            if len(image.shape) == 3:  # Assuming (H, W, C) = (128, 128, 8)
                # Permute to (C, H, W) then add two batch dimensions
                image = image.permute(2, 0, 1)       # (8, 128, 128)
                image = image.unsqueeze(0)           # (1, 8, 128, 128)
                image = image.unsqueeze(0)           # (1, 1, 8, 128, 128)

            # Move tensors them to the correct device
            image = image.to(self.device)
            
            # Generate predictions
            prediction = self.model(image)

            prediction = prediction.detach().cpu().numpy()
            
            return prediction
