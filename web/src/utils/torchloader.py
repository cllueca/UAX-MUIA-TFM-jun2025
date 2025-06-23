# Builtins
import gc
# Installed
import torch
# Local
from src.unet import (
    UNet3D,
    AttentionUnet3D
)
# Types
from typing import Any


# Load the PyTorch model
def load_torch_model_state_dict(
    model_path: str,
    model_architecture: UNet3D | AttentionUnet3D,
    device: str='cuda' if torch.cuda.is_available() else 'cpu'
) -> UNet3D | AttentionUnet3D:
    """
    Load a saved PyTorch model.
    
    Args:
        model_path: Path to the saved model state dictionary
        model_architecture: The model architecture instance
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Loaded PyTorch model
    """
    # Load the state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
    except:
        state_dict = torch.load(model_path.replace('pt', 'pth'), map_location=device)
    
    # Apply the state dictionary to the model
    model_architecture.load_state_dict(state_dict)
    
    # Move model to the specified device
    model_architecture = model_architecture.to(device)
    
    # Set to evaluation mode
    model_architecture.eval()
    
    return model_architecture


# Load the complete model:
def load_complete_model(model_path: str) -> UNet3D | AttentionUnet3D:
    """Load a complete model including architecture and weights"""
    try:
        model = torch.load(model_path, weights_only=False, map_location='cuda')
    except:
        model = torch.load(model_path.replace('pt', 'pth'), weights_only=False, map_location='cuda')
    model.eval()  # Set to evaluation mode
    return model


# Function to load a model saved with parameters
def load_model_with_params(
    model_path: str,
    model_config: dict[str, Any],
    device: str=None
) -> UNet3D | AttentionUnet3D:
    """
    Load a model that was saved with its class name and initialization parameters
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model to (default: None, will use the same device as saved)
    
    Returns:
        model: The loaded model
        model_info: Dictionary containing model metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved dictionary
    model_info = torch.load(model_path, map_location=device)
    
    # Get the class name and parameters
    class_name = model_info['class_name']
    # init_params = model_info['init_params']
    init_params = {
        "model_config": model_config
    }
    
    # Dynamically create an instance of the model class
    if class_name in ['UNet3D', 'AttentionUnet3D']:
        if model_config['model_type'] == 'unet-original':
            model = UNet3D(**init_params)
        elif model_config['model_type'] == 'unet-attention':
            model = AttentionUnet3D(**init_params)
    else:
        raise ValueError(f"Unknown model class: {class_name}")
    
    # Load the state dict
    model.load_state_dict(model_info['state_dict'])
    
    # Move model to the specified device
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, model_info
