"""
Logging Configuration Module
=============================
Comprehensive logging setup for production ML pipelines.

This module provides:
- File and console logging
- Log rotation
- Structured logging with context
- Debug information tracking
- Error reporting with stack traces
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
from logging.handlers import RotatingFileHandler
import traceback
import json


# Project root path
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class StructuredLogger:
    """
    Custom logger with structured logging capabilities.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = get_logger(name, level)
        self.context: Dict[str, Any] = {}

    def add_context(self, **kwargs):
        """Add context to logger."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear logger context."""
        self.context = {}

    def _format_message(self, message: str) -> str:
        """Format message with context."""
        if self.context:
            context_str = json.dumps(self.context)
            return f"{message} | Context: {context_str}"
        return message

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.add_context(**kwargs)
        self.logger.debug(self._format_message(message))

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.add_context(**kwargs)
        self.logger.info(self._format_message(message))

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.add_context(**kwargs)
        self.logger.warning(self._format_message(message))

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        self.add_context(**kwargs)
        if exc_info:
            self.logger.error(
                f"{self._format_message(message)}\n{traceback.format_exc()}"
            )
        else:
            self.logger.error(self._format_message(message))

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.add_context(**kwargs)
        self.logger.critical(self._format_message(message))


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

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler - detailed logs (all levels)
    log_file = LOGS_DIR / f'training_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10 MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Error file handler - errors only
    error_file = LOGS_DIR / f'errors_{datetime.now().strftime("%Y%m%d")}.log'
    error_handler = RotatingFileHandler(
        error_file, maxBytes=5 * 1024 * 1024, backupCount=3  # 5 MB
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


class PipelineLogger:
    """
    Logger specifically designed for ML pipelines.
    Tracks training progress, metrics, and issues.
    """

    def __init__(self, name: str = "pipeline"):
        self.logger = get_logger(name)
        self.metrics_history: list = []
        self.start_time: Optional[datetime] = None

    def log_pipeline_start(self, config: Dict[str, Any]):
        """Log pipeline start with configuration."""
        self.start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def log_pipeline_end(self, status: str = "SUCCESS"):
        """Log pipeline end with status."""
        duration = datetime.now() - self.start_time if self.start_time else None
        self.logger.info("=" * 60)
        self.logger.info(f"PIPELINE {status}")
        if duration:
            self.logger.info(f"Duration: {duration}")
        self.logger.info("=" * 60)

    def log_stage(self, stage_name: str):
        """Log pipeline stage."""
        self.logger.info(f"--- Starting Stage: {stage_name} ---")

    def log_metrics(self, stage: str, metrics: Dict[str, float]):
        """Log metrics for a stage."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{stage}] Metrics: {metrics_str}")
        self.metrics_history.append(
            {
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            }
        )

    def log_data_info(self, stage: str, shape: tuple, columns: list = None):
        """Log data information."""
        self.logger.info(f"[{stage}] Data shape: {shape}")
        if columns:
            self.logger.debug(f"[{stage}] Columns: {columns}")

    def log_model_info(self, model_name: str, params: Dict[str, Any]):
        """Log model information."""
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")

    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")


# Default logger instance
default_logger = get_logger(__name__)


def log_function_call(func):
    """
    Decorator to log function calls.
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def log_execution_time(func):
    """
    Decorator to log function execution time.
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            duration = datetime.now() - start_time
            logger.info(f"{func.__name__} completed in {duration}")
            return result
        except Exception as e:
            duration = datetime.now() - start_time
            logger.error(f"{func.__name__} failed after {duration}: {str(e)}")
            raise

    return wrapper
