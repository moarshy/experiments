import os
from dotenv import load_dotenv
import logging
load_dotenv() # Load variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o-mini"
LOG_LEVEL = logging.INFO
UNIQUE_VALUE_THRESHOLD = 20