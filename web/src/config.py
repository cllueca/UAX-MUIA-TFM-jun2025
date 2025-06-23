# Builtins
from pathlib import Path
import os
# Installed
from dotenv import dotenv_values
from pydantic import BaseModel
# Local
# Types
from typing import (
    ClassVar,
    Dict
)


APP_CONFIG = dotenv_values(Path(__file__).parent.parent / 'config/app_config.env')
DB_CONFIG = dotenv_values(Path(__file__).parent.parent / 'config/db_config.env')


class _Config(BaseModel):

    print(f"{'='*50} CONFIGURATION {'='*50}")

    # ANSI escape sequences for colors
    LOGGER_COLORS: ClassVar[Dict[str, str]] = {
        'DEBUG': '\033[94m',            # Blue
        'BOLD_DEBUG': '\033[1;94m',     # Bold blue
        'INFO': '\033[92m',             # Green
        'BOLD_INFO': '\033[1;92m',      # Bold green
        'WARNING': '\033[93m',          # Yellow
        'BOLD_WARNING': '\033[1;93m',   # Bold yellow
        'ERROR': '\033[91m',            # Red
        'BOLD_ERROR': '\033[1;91m',     # Bold red
        'CRITICAL': '\033[41m',         # Red background
        'BOLD': '\033[1m',              # Bold text
        'RESET': '\033[0m',             # Reset to default
    }

    API_HOST: ClassVar[str] = APP_CONFIG['API_HOST']
    API_PORT: ClassVar[int] = int(APP_CONFIG['API_PORT'])

    VISION_NN_FOLDER: ClassVar[Path] = Path(__file__).parent.parent / APP_CONFIG['VISION_NN_FOLDER']

    if not os.path.exists(VISION_NN_FOLDER):
        raise OSError("Neural networks folder not found, could not initialize API")
    
    DB_PATH: ClassVar[Path] = Path(__file__).parent.parent / os.path.join(DB_CONFIG['DB_FOLDER_NAME'], DB_CONFIG['DB_FILE_NAME'])
    DB_KEY_PATH: ClassVar[Path] = Path(__file__).parent.parent / os.path.join(DB_CONFIG['DB_FOLDER_NAME'], APP_CONFIG['ENCRYPTION_KEY_FILENAME'])
    PATIENT_HASH_SLICE: ClassVar[int] = 8

    DISPLAY_INDENTATION: ClassVar[int] = 30

    MODEL_NAME: ClassVar[str] = APP_CONFIG['MODEL_NAME']

    print(f"  - API_HOST{'.'*(DISPLAY_INDENTATION-len('API_HOST'))} {API_HOST}")
    print(f"  - API_PORT{'.'*(DISPLAY_INDENTATION-len('API_PORT'))} {API_PORT}")
    print(f"  - VISION_NN_FOLDER{'.'*(DISPLAY_INDENTATION-len('VISION_NN_FOLDER'))} {VISION_NN_FOLDER}")
    print(f"  - DB_PATH{'.'*(DISPLAY_INDENTATION-len('DB_PATH'))} {DB_PATH}")
    print(f"  - MODEL_NAME{'.'*(DISPLAY_INDENTATION-len('MODEL_NAME'))} {MODEL_NAME}")

    print("="*115, end="\n\n")

CONFIG = _Config