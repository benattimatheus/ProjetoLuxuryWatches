import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs.txt"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("EDA_Project")

logger = setup_logger()
