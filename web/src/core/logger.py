import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import traceback
# Local
from src.config import CONFIG

class CustomFormatter(logging.Formatter):
    """Custom formatter that includes detailed traceback information for errors"""
    COLORS = CONFIG.LOGGER_COLORS

    def format(self, record):
        # Save the original format
        format_orig = self._style._fmt
        levelname_orig = record.levelname
        # Modify the format for errors to include traceback
        if record.levelno >= logging.ERROR and record.exc_info:
            self._style._fmt = "%(levelname)s:\t  %(name)s [%(asctime)s] -- %(message)s\nTraceback:\n%(traceback)s"
            if not hasattr(record, 'traceback'):
                record.traceback = ''.join(traceback.format_exception(*record.exc_info))
        
        # # Add color based on log level on message
        # levelname = record.levelname
        # color = self.COLORS.get(levelname, self.COLORS['RESET'])
        # record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        
        # Add color based on log level on log level
        color = self.COLORS.get(levelname_orig, self.COLORS['RESET'])
        record.levelname = f"{color}{levelname_orig}{self.COLORS['RESET']}"
        # Format the record
        result = logging.Formatter.format(self, record)
        # Restore the original format
        self._style._fmt = format_orig
        record.levelname = levelname_orig
        return result
    
def setup_logger(
    name: str = "TFM-Demo-Core",
    log_dir: str | Path = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024, # 10MB
    backup_count: int = 5,
    console_output: bool = True
) -> logging.Logger:
    """Sets up a logger with rotating file handler and optional console output"""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Create formatter
    formatter = CustomFormatter(
        '%(levelname)s:\t  %(name)s [%(asctime)s] -- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Create rotating file handler
    log_file = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    # Capture unhandled exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default handler for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    sys.excepthook
    return logger

# Create all the necessary logger instances
CORE_LOGGER = setup_logger()
DOCTOR_LOGGER = setup_logger(name="TFM-Demo-Doctor")
PATIENT_LOGGER = setup_logger(name="TFM-Demo-Patient")