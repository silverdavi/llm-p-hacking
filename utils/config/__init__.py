"""
API configuration settings for the project.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_CONFIG = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "defaults": {
            "options_model": "o1-mini",
            "min_delay": 0.1,  # Minimum delay between API calls in seconds
            "max_retries": 3,  # Maximum number of retries for failed API calls
            "timeout": 30.0,   # Default timeout in seconds
        }
    }
} 