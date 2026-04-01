import logging
import sys


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent propagation to the root logger
    logger.propagate = False

    # Create handlers if they don't exist
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
