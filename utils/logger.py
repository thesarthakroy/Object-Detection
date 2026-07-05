"""
Logging configuration module.
Sets up unified formatters and console/file output streams.
"""
import logging
import os

def setup_logger(log_file: str = "app.log", log_level: int = logging.INFO) -> None:
    """Configures system-wide logging with file and console output handlers."""
    # Ensure handlers are reset to prevent duplicate messages
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    logging.info("VMS System logger initialized. Outputs routed to: %s", log_file)
