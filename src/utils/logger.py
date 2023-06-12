import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    name = name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger
