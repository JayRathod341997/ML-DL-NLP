"""Logging utility for KNN Customer Segmentation project"""

import logging
import os
from datetime import datetime


def get_logger(name, log_level=logging.INFO):
    """Get a configured logger instance"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # File handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'knn_{datetime.now().strftime("%Y%m%d")}.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
