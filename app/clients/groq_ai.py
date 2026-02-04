import aiohttp
import json
import logging
import re
from typing import Dict, List, Any, Optional
import os
from app.clients.base import AIClient

class GroqAIClient(AIClient):
    """
    Client for interacting with the Groq AI API to access deepseek models.
    """
    
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "deepseek-r1-distill-llama-70b"

    async def call_llm(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an asynchronous call to the Groq API using specified or default model.
        
        Args:
            messages: List of message dictionaries in chat format
            max_tokens: Maximum tokens for the response (defaults to 4096)
            model: Model name to use (defaults to deepseek-r1-distill-llama-70b)
            
        Returns:
            Dictionary containing the parsed JSON response
        """
        primary_model = model or self.model
        fallback_model = "qwen-2.5-32b" if primary_model == "deepseek-r1-distill-llama-70b" else "deepseek-r1-distill-llama-70b"
        
        result = await self._try_model_call(messages, primary_model, max_tokens)
        if not result:
            logging.info(f"Primary model '{primary_model}' failed, trying fallback model '{fallback_model}'")
            result = await self._try_model_call(messages, fallback_model, max_tokens)
            
        return result
        
    async def _try_model_call(self, messages: List[Dict[str, str]], model: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Attempt a call to a specific model.
        
        Args:
            messages: List of message dictionaries in chat format
            model: Model name to use
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dictionary containing the parsed JSON response or empty dict if failed
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens or 4096,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, headers=headers, json=payload) as response:
                        response.raise_for_status()
                        result = await response.json()
                result = result["choices"][0]["message"]["content"]
                logging.info(f"Response from Groq model '{model}' received")
                
                return await self.parse_json(result)
            except Exception as e:
                logging.warning(f"Error while calling Groq with model '{model}': {str(e)}")
                return {}
            
        except Exception as e:
            logging.warning(f"Error while calling Groq with model '{model}': {str(e)}")
            return {}

    async def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Attempt to parse a JSON response. If parsing fails,
        try various methods to extract valid JSON.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed JSON as a dictionary or empty dict if parsing fails
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            start_index = response.find('{')
            end_index = response.rfind('}')
            if start_index != -1 and end_index != -1:
                json_str = response[start_index:end_index+1]
                try:
                    return json.loads(json_str)
                except Exception:
                    json_str = self.extract_json_from_code_block(response)
                    try:
                        if json_str:
                            start_index = json_str.find('{')
                            end_index = json_str.rfind('}')
                            if start_index != -1 and end_index != -1:
                                json_str = json_str[start_index:end_index+1]
                                return json.loads(json_str)
                    except Exception as e:
                        logging.error(f"Error while parsing json: {str(e)}")
            return {}

    def extract_json_from_code_block(self, response: str) -> Optional[str]:
        """
        Extract JSON from a markdown code block.
        
        Args:
            response: Raw response string that might contain markdown code blocks
            
        Returns:
            Extracted JSON string or None if not found
        """
        pattern = r"```json(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            return json_content
        return None
