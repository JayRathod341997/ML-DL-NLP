"""
Logging Configuration Module
Credit Scoring Project - Decision Tree
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create logs directory
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    # File handler - logs all levels
    log_file = os.path.join(
        log_dir, f'credit_scoring_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)

    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Example usage:
# from src.utils.logger import get_logger
#
# logger = get_logger(__name__)
#
# logger.debug("Debug message - detailed diagnostic information")
# logger.info("Info message - confirmation that things are working")
# logger.warning("Warning message - something unexpected happened")
# logger.error("Error message - serious problem")
# logger.critical("Critical message - program may crash")


# Log Levels Guide:
# -----------------
# DEBUG (10): Detailed diagnostic information
# - Variable values
# - Function execution flow
# - Detailed error stack traces
#
# INFO (20): Confirmation that things are working as expected
# - Successful operations
# - Business milestones
# - System events
#
# WARNING (30): Something unexpected happened, but the program still works
# - Deprecated features
# - Resource constraints
# - Minor data issues
#
# ERROR (40): Serious problem, the program failed to perform some function
# - Exceptions caught
# - Database connection failures
# - Missing files
#
# CRITICAL (50): Very serious error, program may crash
# - Out of memory
# - System shutdown
# - Data corruption


# Production Logging Best Practices:
# --------------------------------
# 1. Use structured logging (JSON format) in production
# 2. Log sensitive data cautiously (PII, passwords)
# 3. Include correlation IDs for request tracing
# 4. Set appropriate log rotation policies
# 5. Use different log levels for different environments
