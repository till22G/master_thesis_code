import logging

def _create_logger():
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    return logger   

logger = _create_logger()
