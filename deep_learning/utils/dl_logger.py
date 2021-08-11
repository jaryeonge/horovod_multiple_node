import logging


def default_logger():
    ''' This function returns the pre-defined logger '''
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel('INFO')
    formatter = logging.Formatter('%(asctime)s: %(levelname)s >> %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(stream_handler)
    return logger
