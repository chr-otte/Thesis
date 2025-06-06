import logging 

def configure_logging():
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplication
    while root_logger.handlers:
        root_logger.handlers.pop()
    
    # Add a console handler for our logs
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Disable Flower's default logging
    flower_logger = logging.getLogger("flwr")
    flower_logger.propagate = False  # Don't propagate to root logger
    return logging.getLogger("FL-Experiment")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FL-Experiment")
logger = configure_logging()
