import os
import sys
import logging

log_dir = "logs"
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_filepath = os.path.join(log_dir, "continious_logs.log")

os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers=[  ## Here we are using two handlers
        logging.FileHandler(log_filepath),  ## Where we are writing all the details in the log_filepath
        logging.StreamHandler(sys.stdout)   ## StreamHandler which will display all the logs in my output terminal
    ]
)

logger = logging.getLogger("summarizerlogger")