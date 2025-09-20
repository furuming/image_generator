import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present,
# searching up from the current working directory.
load_dotenv()

# Exported settings
APP_PORT = int(os.getenv("APP_PORT", "5000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")
