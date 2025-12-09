"""
Logging configuration utility.
"""
import logging
import sys


def setup_logger(name=__name__, level=logging.INFO, debug=False):
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        debug: If True, sets level to DEBUG and adds more verbose formatting
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set logging level
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Set handler level
    if debug:
        handler.setLevel(logging.DEBUG)
    else:
        handler.setLevel(level)
    
    # Create formatter
    if debug:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

