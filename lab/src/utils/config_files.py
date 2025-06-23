# Builtins
import json
from pathlib import Path
# Types
from typing import (
    Dict,
    Any
)

def save_config(
    json_path: Path,
    json_data: Dict[str, Any],
    calling_class=None
) -> None:
    """
    Save the given configuration data to a JSON file.

    Args:
        json_path (Path): The file path where the JSON data will be saved.
        json_data (Dict[str, Any]): The configuration data to save.
        calling_class (function): Class calling this function
            Default: None

    Returns:
        None
    """        
    try:
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        if calling_class is not None:
            calling_class._log(f"Configuration saved to: '{json_path}'...")
        else:
            print((f"Configuration saved to: '{json_path}'..."))
    except IOError as e:
        if calling_class is not None:
            calling_class._log(f"An error occurred while writing to the file: {e}")
        else:
            raise IOError(f"An error occurred while writing to the file: {e}")
    except Exception as e:
        if calling_class is not None:
            calling_class._log(f"An unexpected error occurred saving the configuration: {e}")
        else:
            raise Exception(f"An unexpected error occurred saving the configuration: {e}")



def load_config(
    json_path: Path,
    calling_class=None
) -> Dict[str, Any]:
    """
    Load a configuration from a given JSON file.

    Args:
        json_path (Path): The file path where the JSON data will be saved.
        calling_class (function): Class calling this function
            Default: None

    Returns:
        None
    """    
    try:
        with open(json_path, 'r') as json_file:
            loaded_config = json.load(json_file)
        if calling_class is not None:
            calling_class(f"Configuration from '{json_path}' loaded successfully.")
        else:
            print(f"Configuration from '{json_path}' loaded successfully.")
    except FileNotFoundError:
        if calling_class is not None:
            calling_class._log(f"The file {json_path} does not exist.")
        else:
            raise FileNotFoundError(f"The file {json_path} does not exist.")
    except json.JSONDecodeError:
        if calling_class is not None:
            calling_class._log(f"Error decoding JSON from the file {json_path}.")
        else:
            raise json.JSONDecodeError(f"Error decoding JSON from the file {json_path}.")
    except Exception as e:
        if calling_class is not None:
            calling_class._log(f"An unexpected error occurred: {e}")
        else:
            raise Exception(f"An unexpected error occurred: {e}")

    return loaded_config
