from logging import getLogger, INFO, FileHandler, Formatter

# As the first step, a logger is created using the Python logging module

def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler = FileHandler(filename = f'{filename}.txt')
    handler.setFormatter(Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger
