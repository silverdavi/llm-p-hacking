"""
Parallel processing wrapper for LLM API calls with rate limiting and retry functionality.
Uses asyncio for concurrent API calls while respecting rate limits.
"""

import asyncio
import time
import logging
import aiohttp
from typing import List, Dict, Any, Optional, Union, Tuple
from utils.config import API_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Define which models use chat completions endpoint
CHAT_MODELS = {
    "gpt-4o",
    "gpt-4.5-preview",
    "chatgpt-4o-latest",
    "o1",
    "o3-mini-2025-01-31",
    "o1-mini"
}

# Define which models don't support system messages
NO_SYSTEM_MODELS = {
    "o1-mini"
}

# Define which models use response endpoint
RESPONSE_MODELS = {
    "o1-pro"
}

# Define which models don't support temperature
NO_TEMPERATURE_MODELS = {
    "o1-mini"
}

# Define which models use max_tokens vs max_completion_tokens
MAX_COMPLETION_TOKENS_MODELS = {
    "o1-mini",
    "o1"
}

# Define API endpoints for different model types
API_ENDPOINTS = {
    "o1-mini": "https://api.openai.com/v1/chat/completions",
    "gpt-4o": "https://api.openai.com/v1/chat/completions",
    "default": "https://api.openai.com/v1/completions"
}

class ParallelLLM:
    """Wrapper for OpenAI API with parallel processing and rate limiting"""

    def __init__(
            self,
            api_key: str,
            model: str = API_CONFIG["openai"]["defaults"]["options_model"],
            max_concurrent: int = 5,
            min_delay: float = API_CONFIG["openai"]["defaults"]["min_delay"],
            max_retries: int = API_CONFIG["openai"]["defaults"]["max_retries"],
            retry_delay: float = 1.0
    ):
        """
        Initialize parallel LLM API wrapper.

        Args:
            api_key: OpenAI API key
            model: Model to use for completions
            max_concurrent: Maximum number of concurrent API calls
            min_delay: Minimum delay between API calls in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_call_time = 0
        self.total_calls = 0
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Determine endpoint based on model
        self.endpoint = API_ENDPOINTS.get(model, API_ENDPOINTS["default"])
        self.is_chat_model = model in CHAT_MODELS
        self.is_response_model = model in RESPONSE_MODELS
        self.supports_temperature = model not in NO_TEMPERATURE_MODELS
        self.uses_max_completion_tokens = model in MAX_COMPLETION_TOKENS_MODELS

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """Make a single API call with retry logic and rate limiting."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_delay:
            await asyncio.sleep(self.min_delay - time_since_last_call)
        
        for attempt in range(self.max_retries):
            try:
                # Create request payload based on model type
                if self.is_response_model:
                    payload = {
                        "model": self.model,
                        "input": prompt,
                        "max_output_tokens": 2048
                    }
                elif self.is_chat_model:
                    if self.model in NO_SYSTEM_MODELS:
                        # For models that don't support system messages, include instructions in user message
                        payload = {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"""You are a scientific researcher analyzing correlations in data. Provide detailed, evidence-based explanations.

{prompt}"""
                                }
                            ]
                        }
                        
                        # Use appropriate max tokens parameter
                        if self.uses_max_completion_tokens:
                            payload["max_completion_tokens"] = 1000
                        else:
                            payload["max_tokens"] = 1000
                            
                        # Add temperature if supported
                        if self.supports_temperature:
                            payload["temperature"] = 0.7
                    else:
                        # For models that support system messages
                        payload = {
                            "model": self.model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are a scientific researcher analyzing correlations in data. Provide detailed, evidence-based explanations."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ]
                        }
                        
                        # Use appropriate max tokens parameter
                        if self.uses_max_completion_tokens:
                            payload["max_completion_tokens"] = 1000
                        else:
                            payload["max_tokens"] = 1000
                        
                        # Add temperature if supported
                        if self.supports_temperature:
                            payload["temperature"] = 0.7
                else:
                    payload = {
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 1000
                    }
                    
                    # Add temperature if supported
                    if self.supports_temperature:
                        payload["temperature"] = 0.7
                
                self.logger.debug(f"Making API call to {self.endpoint} with payload: {payload}")
                
                async with self.session.post(
                    self.endpoint,
                    json=payload
                ) as response:
                    response_text = await response.text()
                    self.logger.debug(f"API response: {response_text}")
                    
                    if response.status == 200:
                        self.last_call_time = time.time()
                        self.total_calls += 1
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        retry_after = float(response.headers.get("Retry-After", self.retry_delay))
                        self.logger.warning(f"Rate limited. Waiting {retry_after}s before retry.")
                        await asyncio.sleep(retry_after)
                    else:
                        self.logger.error(f"API call failed with status {response.status}: {response_text}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                        else:
                            raise Exception(f"API call failed after {self.max_retries} attempts: {response_text}")
            except Exception as e:
                self.logger.error(f"API call failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of prompts in parallel while respecting rate limits."""
        if not self.session:
            raise RuntimeError("ParallelLLM must be used as an async context manager")
            
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(prompt: str) -> Dict[str, Any]:
            async with semaphore:
                return await self._make_api_call(prompt)
        
        tasks = [process_with_semaphore(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return {
            "total_calls": self.total_calls,
            "last_call_time": self.last_call_time
        } 