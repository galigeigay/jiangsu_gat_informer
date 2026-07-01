import logging

def setup_logger(name, level=logging.INFO, logfile = None):
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(level=level)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger