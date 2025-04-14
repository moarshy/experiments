import logging
from insights.config import LOG_LEVEL

def setup_logging(log_level: logging.Logger = LOG_LEVEL):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
