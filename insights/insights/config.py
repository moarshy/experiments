import os
from dotenv import load_dotenv
import logging
load_dotenv() # Load variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SQL_LLM_PROVIDER = "gemini"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 5000
DEFAULT_THINKING_BUDGET = 5000
DEFAULT_MAX_RETRIES = 3
LOG_LEVEL = logging.INFO
UNIQUE_VALUE_THRESHOLD = 20
DB_CONFIG = {
    "type": "sqlite",
    "database": "/Users/arshath/play/experiments/insights/tests/it_sales.db"
}
MAX_CONCURRENT_WORKERS = 10
VANNA_COLLECTION_NAME = "insights_db"
VANNA_PERSIST_DIR = "./vanna_data"