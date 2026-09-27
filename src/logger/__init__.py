import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Configuration Constants
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(filename)s - %(name)s - %(levelname)s - %(message)s"

def get_logger(name: str = "mlops") -> logging.Logger:
    """Configures and returns a logger instance with non-duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent attaching handlers multiple times on module re-import
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        # File Handler (DEBUG level)
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console Handler (INFO level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Default logger instance
logger = get_logger("mlops_pipeline")