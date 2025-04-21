"""
Utility functions for making LLM API calls.
Provides a simplified interface for interacting with LLM APIs.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from utils.api.llm_api import LLMApi
from utils.config import API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients (lazy initialization)
_llm_client = None


def get_llm_client(
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        min_delay: Optional[float] = None,
        max_retries: Optional[int] = None
) -> LLMApi:
    """
    Get or initialize the OpenAI LLM client.
    
    Args:
        model: OpenAI model to use (optional, defaults to config)
        api_key: API key to use (optional, defaults to config)
        min_delay: Minimum delay between calls (optional, defaults to config)
        max_retries: Maximum number of retries (optional, defaults to config)

    Returns:
        Initialized LLMApi instance
    """
    global _llm_client
    
    # Get defaults from config
    openai_config = API_CONFIG.get("openai", {})
    
    # If no client exists or model has changed, create a new one
    if _llm_client is None or (model and _llm_client.model != model):
        _llm_client = LLMApi(
            api_key=api_key,
            model=model,
            min_delay=min_delay,
            max_retries=max_retries
        )
    
    return _llm_client


def call_openai(
        prompt: Union[str, Dict[str, Any]],
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        return_headers: bool = False
) -> Union[str, Dict[str, Any]]:
    """
    Call OpenAI API with the given prompt, handling different prompt formats.

    Args:
        prompt: Either a string prompt or a dictionary with structured prompt information
               If a dictionary, it should contain 'system', 'user', and optionally 'response_format'
        model: OpenAI model to use (optional, defaults to config)
        timeout: Request timeout in seconds (optional)
        return_headers: Whether to return response headers along with the response

    Returns:
        Response text from the model, or dictionary with response and headers if return_headers is True
    """
    # Get defaults from config if not provided
    if model is None:
        model = API_CONFIG.get("openai", {}).get("defaults", {}).get("options_model", "o1")
    
    client = get_llm_client(model=model)

    try:
        # Handle different prompt formats
        if isinstance(prompt, dict):
            # Extract components from the structured prompt
            system_content = prompt.get('system', '')
            user_content = prompt.get('user', '')
            response_format_str = prompt.get('response_format', None)

            # Create message list
            messages = []
            if system_content:
                messages.append({"role": "system", "content": system_content})
            if user_content:
                messages.append({"role": "user", "content": user_content})

            # Process response_format - convert string to proper object format if needed
            response_format = None
            if response_format_str:
                if isinstance(response_format_str, str) and response_format_str.lower() == 'json':
                    response_format = {"type": "json_object"}
                elif isinstance(response_format_str, dict):
                    response_format = response_format_str
                else:
                    logger.warning(f"Ignoring invalid response_format: {response_format_str}")

            # Call LLMApi with structured format
            response = client.call_structured_model(
                messages=messages,
                response_format=response_format,
                timeout=timeout,
                return_headers=return_headers
            )
        else:
            # Handle simple string prompt
            response = client.call_model(prompt, timeout=timeout, return_headers=return_headers)
            
        if return_headers:
            return response
        return response.get('response', '') if isinstance(response, dict) else response
            
    except Exception as e:
        logger.error(f"Error calling OpenAI ({model}): {e}")
        raise 