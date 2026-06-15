import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("frauddetection")
# Affiche aussi les logs dans la console (pratique pendant la démo)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))
logger.addHandler(_console)
