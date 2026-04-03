import logging
from ..constants import LOG_FOLDER

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(LOG_FOLDER + "/" + name + ".log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(level)

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger