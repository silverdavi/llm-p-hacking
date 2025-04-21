"""
Utility script to check API rate limits.
This provides a lightweight way to check rate limits without running the full experiment.
"""

import logging
from utils.api.util_call import call_openai
from utils.config import API_CONFIG

def check_rate_limits(model: str = None) -> dict:
    """
    Check API rate limits by making a minimal API call.
    
    Args:
        model: Optional model name to check limits for. Defaults to config.
        
    Returns:
        dict: Rate limit information
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Checking API rate limits...")
    
    # Use default model from config if none specified
    if model is None:
        model = API_CONFIG.get("openai", {}).get("defaults", {}).get("options_model", "o1-mini")
    
    # Create a minimal prompt
    test_prompt = "Hello, this is a test message to check rate limits."
    
    try:
        # Make the API call with return_headers=True to get rate limit info
        response = call_openai(
            prompt=test_prompt,
            model=model,
            return_headers=True
        )
        
        # Extract rate limits from headers
        headers = response.get('headers', {})
        
        # Print rate limit information
        print("\nDetailed Rate Limit Information:")
        print("================================")
        print(f"Model: {model}")
        print(f"Requests Limit: {headers.get('x-ratelimit-limit-requests', 'N/A')}")
        print(f"Remaining Requests: {headers.get('x-ratelimit-remaining-requests', 'N/A')}")
        print(f"Requests Reset Time: {headers.get('x-ratelimit-reset-requests', 'N/A')}")
        print(f"Tokens Limit: {headers.get('x-ratelimit-limit-tokens', 'N/A')}")
        print(f"Remaining Tokens: {headers.get('x-ratelimit-remaining-tokens', 'N/A')}")
        print(f"Tokens Reset Time: {headers.get('x-ratelimit-reset-tokens', 'N/A')}")
        
        return headers
        
    except Exception as e:
        logger.error(f"Error checking rate limits: {e}")
        return {}

if __name__ == "__main__":
    check_rate_limits() 