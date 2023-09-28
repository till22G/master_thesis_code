import logging

def _create_logger():
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s "))
    logger.addHandler(console_handler)
    
    return logger   

logger = _create_logger()