import aiohttp
import json
import logging
import re
import asyncio
import time
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
        # Using llama-3.1-8b-instant (production model, fast and reliable)
        self.model = "llama-3.1-8b-instant"

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
        # Fallback to llama-3.3-70b-versatile (more powerful) if 8b fails
        fallback_model = "llama-3.3-70b-versatile" if "8b" in primary_model else "llama-3.1-8b-instant"
        
        result = await self._try_model_call(messages, primary_model, max_tokens)
        if not result:
            logging.info(f"Primary model '{primary_model}' failed, trying fallback model '{fallback_model}'")
            result = await self._try_model_call(messages, fallback_model, max_tokens)
            
        return result
        
    async def _try_model_call(self, messages: List[Dict[str, str]], model: str, max_tokens: Optional[int] = None, retry_count: int = 0) -> Dict[str, Any]:
        """
        Attempt a call to a specific model with retry logic for rate limits.
        
        Args:
            messages: List of message dictionaries in chat format
            model: Model name to use
            max_tokens: Maximum tokens for the response
            retry_count: Current retry attempt (for exponential backoff)
            
        Returns:
            Dictionary containing the parsed JSON response or empty dict if failed
        """
        max_retries = 3
        base_delay = 5  # Start with 5 seconds for Groq
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Ensure first message instructs JSON output if not already present
            formatted_messages = self._ensure_json_instruction(messages)
            
            payload = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or 8192,  # Increased for better responses
                "temperature": 0.1,  # Lower temperature for more consistent JSON
            }
            
            # Try to use structured outputs if available (Groq supports this)
            # Check if model supports structured outputs
            if "llama-3.3" in model or "llama-3.1" in model:
                try:
                    payload["response_format"] = {"type": "json_object"}
                except:
                    pass  # If not supported, continue without it

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.url, headers=headers, json=payload) as response:
                        # Check for rate limit errors
                        if response.status == 429:
                            if retry_count < max_retries:
                                delay = base_delay * (2 ** retry_count)  # 5s, 10s, 20s
                                logging.warning(f"Rate limit hit (429). Retrying in {delay}s (attempt {retry_count + 1}/{max_retries})...")
                                await asyncio.sleep(delay)
                                return await self._try_model_call(messages, model, max_tokens, retry_count + 1)
                            else:
                                logging.error(f"Rate limit exceeded after {max_retries} retries. Please wait before trying again.")
                                return {"error": "Rate limit exceeded. Please wait a few minutes and try again."}
                        
                        if response.status != 200:
                            error_body = await response.text()
                            logging.error(f"Groq API returned {response.status}: {error_body}")
                            return {}
                        
                        result = await response.json()
                        if "choices" not in result or not result["choices"]:
                            logging.error(f"Unexpected Groq response format: {result}")
                            return {}
                        
                        content = result["choices"][0]["message"]["content"]
                        logging.info(f"Response from Groq model '{model}' received ({len(content)} chars)")
                        
                        return await self.parse_json(content)
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        logging.warning(f"Rate limit hit (429). Retrying in {delay}s (attempt {retry_count + 1}/{max_retries})...")
                        await asyncio.sleep(delay)
                        return await self._try_model_call(messages, model, max_tokens, retry_count + 1)
                    else:
                        logging.error(f"Rate limit exceeded after {max_retries} retries.")
                        return {"error": "Rate limit exceeded. Please wait a few minutes and try again."}
                # Log the actual error response for debugging
                try:
                    error_body = await e.response.json()
                    logging.error(f"Groq API error ({e.status}): {error_body}")
                except:
                    logging.error(f"Groq API error ({e.status}): {str(e)}")
                return {}
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if retry_count < max_retries:
                        delay = base_delay * (2 ** retry_count)
                        logging.warning(f"Rate limit detected. Retrying in {delay}s (attempt {retry_count + 1}/{max_retries})...")
                        await asyncio.sleep(delay)
                        return await self._try_model_call(messages, model, max_tokens, retry_count + 1)
                logging.warning(f"Error while calling Groq with model '{model}': {str(e)}")
                return {}
            
        except Exception as e:
            logging.warning(f"Error while calling Groq with model '{model}': {str(e)}")
            return {}

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean JSON string by removing control characters and fixing common issues.
        
        Args:
            json_str: Raw JSON string that may contain issues
            
        Returns:
            Cleaned JSON string
        """
        if not json_str:
            return json_str
        
        # Strategy: Process the JSON string more carefully
        # First, remove truly problematic control characters
        json_str = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # Try to fix unescaped newlines/tabs/carriage returns inside string values
        # We'll use a state machine approach: track if we're inside a string
        result = []
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
            elif char == '\\':
                result.append(char)
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
                result.append(char)
                escape_next = False  # Reset escape flag after quote
            elif in_string:
                # Inside a string - escape control characters
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                else:
                    result.append(char)
            else:
                # Outside string - keep as is
                result.append(char)
            
            i += 1
        
        cleaned = ''.join(result)
        
        # Fix common escape issues - unescaped backslashes that aren't part of valid escapes
        cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu0-9x])', r'\\\\', cleaned)
        
        return cleaned
    
    async def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Attempt to parse a JSON response. If parsing fails,
        try various methods to extract valid JSON.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed JSON as a dictionary or empty dict if parsing fails
        """
        if not response or not isinstance(response, str):
            return {}
        
        # Try direct parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try cleaning and parsing
        try:
            cleaned = self._clean_json_string(response)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Extract JSON from code blocks
        json_str = self.extract_json_from_code_block(response)
        if json_str:
            try:
                cleaned = self._clean_json_string(json_str)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object boundaries
        start_index = response.find('{')
        end_index = response.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = response[start_index:end_index+1]
            try:
                cleaned = self._clean_json_string(json_str)
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                # Try to fix common issues and parse again
                try:
                    # Remove trailing commas before closing braces/brackets
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Fix single quotes to double quotes (common mistake)
                    fixed_json = re.sub(r"'([^']*)'", r'"\1"', fixed_json)
                    cleaned = self._clean_json_string(fixed_json)
                    return json.loads(cleaned)
                except Exception as parse_error:
                    logging.error(f"Error while parsing json: {str(e)}")
                    logging.debug(f"Parse error details: {str(parse_error)}")
                    logging.debug(f"Problematic JSON snippet (first 500 chars): {json_str[:500]}")
        
        # Last resort: try to extract any valid JSON fragments or return raw text
        logging.warning("Failed to parse JSON response after all attempts")
        logging.debug(f"Response length: {len(response)} chars")
        logging.debug(f"First 500 chars: {response[:500]}")
        
        # Try one more time with aggressive cleaning
        try:
            # Remove everything before first { and after last }
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                raw_json = response[start:end+1]
                # Aggressive cleaning: remove all control chars, fix escapes
                cleaned = re.sub(r'[\x00-\x1F\x7F]', '', raw_json)  # Remove all control chars
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)  # Remove trailing commas
                # Try to fix common quote issues
                cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
                return json.loads(cleaned)
        except:
            pass
        
        # If all else fails, try to extract text content and return as a simple structure
        # This at least allows the UI to show something
        logging.error("Complete JSON parse failure - returning empty dict")
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
    
    def _ensure_json_instruction(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Ensure the messages include an instruction to return JSON format.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            List of messages with JSON instruction added if needed
        """
        # Check if first message already has JSON instruction
        has_json_instruction = False
        for msg in messages:
            content = msg.get("content", "").lower()
            if "json" in content and ("must" in content or "only" in content or "format" in content):
                has_json_instruction = True
                break
        
        if not has_json_instruction:
            # Add system message instructing JSON output with strict rules
            json_instruction = {
                "role": "system",
                "content": """You MUST respond with valid JSON only. Critical rules:
1. Output ONLY valid JSON - no text before or after
2. Escape all special characters properly (use \\n for newlines, \\" for quotes)
3. Do NOT include control characters (newlines, tabs) in string values - escape them
4. Ensure all strings are properly quoted
5. No trailing commas
6. All keys must be quoted
7. Use double quotes for strings, not single quotes"""
            }
            return [json_instruction] + messages
        
        return messages
