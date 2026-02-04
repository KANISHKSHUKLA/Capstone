import logging
import json
import re
import time
import asyncio
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from app.clients.base import AIClient
from app.config import get_settings

logger = logging.getLogger(__name__)


class GeminiClient(AIClient):
    """
    Client for interacting with Google Gemini AI API.
    """
    
    def __init__(self):
        try:
            super().__init__()
            self.settings = get_settings()
            
            # Only initialize client if API key is available
            if self.settings.GEMINI_API_KEY:
                self.client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
            else:
                logger.warning("Gemini API key not configured. LLM calls will fail.")
                self.client = None
            
            self.model = self.settings.GEMINI_MODEL
            self.fallback_model = self.settings.GEMINI_FALLBACK_MODEL
            self.google_search_tool = types.Tool(googleSearch=types.GoogleSearch())
            self.max_tokens = self.settings.MAX_TOKENS
            self.temperature = self.settings.TEMPERATURE
            self.thinking_budget = self.settings.THINKING_BUDGET
            self.use_grounding = self.settings.USE_GROUNDING
            self.use_thinking = self.settings.USE_THINKING
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise

    async def call_llm(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an asynchronous call to the Gemini API using specified or default model.
        
        Args:
            messages: List of message dictionaries in chat format
            max_tokens: Maximum tokens for the response (defaults to configured max_tokens)
            model: Model name to use (defaults to configured model)
            
        Returns:
            Dictionary containing the parsed JSON response
        """
        if not self.client:
            logger.error("Gemini client not initialized. API key is missing.")
            return {"error": "Gemini API key not configured"}
        
        try:
            primary_model = model or self.model
            fallback_model = self.fallback_model
            
            result = await self._try_model_call(messages, primary_model, max_tokens)
            if not result:
                logger.info(f"Primary model '{primary_model}' failed, trying fallback model '{fallback_model}'")
                result = await self._try_model_call(messages, fallback_model, max_tokens)
                
            return result
        except Exception as e:
            logger.error(f"Error in call_llm: {str(e)}")
            return {}

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
        if not self.client:
            logger.error("Gemini client not initialized")
            return {}
        
        max_retries = 5  # Increased retries
        base_delay = 15  # Start with 15 seconds (longer for free tier)
        
        try:
            start_time = time.time()
            
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                try:
                    role = "user" if msg.get("role") == "user" else "model"
                    content_text = msg.get("content", "")
                    
                    # Handle system messages by prepending to first user message
                    if msg.get("role") == "system":
                        role = "user"
                        content_text = f"SYSTEM INSTRUCTIONS: {content_text}\n\n"
                    
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=content_text)]
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing message: {str(e)}")
                    continue
            
            if not contents:
                logger.error("No valid messages to process")
                return {}
            
            # Build config params
            config_params = {
                "temperature": self.temperature,
                "max_output_tokens": max_tokens or self.max_tokens,
                "response_modalities": ["TEXT"],
            }
            
            # Add grounding if enabled
            if self.use_grounding:
                config_params["tools"] = [self.google_search_tool]
            
            # Add thinking config if enabled
            if self.use_thinking:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            
            config = types.GenerateContentConfig(**config_params)
            
            try:
                # Generate content
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"Response from Gemini model '{model}' received in {elapsed_time:.2f}s")
                logger.debug(f"Response length: {len(response.text)} characters")
                
                return await self.parse_json(response.text)
            except Exception as e:
                error_str = str(e)
                # Check if it's a rate limit error (429)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    if retry_count < max_retries:
                        # Exponential backoff: 15s, 30s, 60s, 120s, 240s
                        delay = base_delay * (2 ** retry_count)
                        logger.warning(f"Rate limit hit (429). Retrying in {delay}s (attempt {retry_count + 1}/{max_retries})...")
                        await asyncio.sleep(delay)
                        return await self._try_model_call(messages, model, max_tokens, retry_count + 1)
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries. Please wait 5-10 minutes before trying again.")
                        return {"error": "Rate limit exceeded. Please wait a few minutes and try again."}
                else:
                    logger.warning(f"Error while calling Gemini with model '{model}': {str(e)}")
                    return {}
            
        except Exception as e:
            logger.warning(f"Error in _try_model_call with model '{model}': {str(e)}")
            return {}

    def fix_json_escaping(self, json_str: str) -> str:
        """
        Fix common JSON escaping issues, particularly with LaTeX backslashes.
        
        Args:
            json_str: Raw JSON string that might have escaping issues
            
        Returns:
            Fixed JSON string with proper escaping
        """
        try:
            # This is a delicate operation - we need to fix LaTeX backslashes without breaking other things
            # Strategy: Replace unescaped backslashes that look like LaTeX commands
            import re
            
            # Common LaTeX commands that need double backslashes in JSON
            latex_commands = [
                'text', 'sigma', 'Sigma', 'rightarrow', 'leftarrow', 'overrightarrow', 'overleftarrow',
                'cdot', 'odot', 'sum', 'prod', 'int', 'frac', 'sqrt', 'alpha', 'beta', 'gamma', 'delta',
                'epsilon', 'theta', 'lambda', 'mu', 'pi', 'tau', 'phi', 'omega', 'partial', 'nabla',
                'infty', 'times', 'div', 'pm', 'mp', 'leq', 'geq', 'neq', 'approx', 'sim', 'propto',
                'forall', 'exists', 'in', 'subset', 'subseteq', 'cap', 'cup', 'emptyset', 'mathbb',
                'mathcal', 'mathbf', 'mathrm', 'log', 'ln', 'exp', 'sin', 'cos', 'tan', 'lim', 'max', 'min'
            ]
            
            # Fix single backslashes before LaTeX commands
            for cmd in latex_commands:
                # Match \command but not \\command (already escaped)
                pattern = r'(?<!\\)\\' + cmd + r'\b'
                json_str = re.sub(pattern, r'\\\\' + cmd, json_str)
            
            # Fix unescaped backslashes in common LaTeX contexts
            # Fix \{ and \} (but not \\{ and \\})
            json_str = re.sub(r'(?<!\\)\\{', r'\\\\{', json_str)
            json_str = re.sub(r'(?<!\\)\\}', r'\\\\}', json_str)
            
            return json_str
        except Exception as e:
            logger.warning(f"Error fixing JSON escaping: {str(e)}")
            return json_str

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
            try:
                # Try to extract JSON from response
                start_index = response.find('{')
                end_index = response.rfind('}')
                if start_index != -1 and end_index != -1:
                    json_str = response[start_index:end_index+1]
                    
                    # Try with escape fixing
                    try:
                        fixed_json = self.fix_json_escaping(json_str)
                        return json.loads(fixed_json)
                    except Exception:
                        # Try original
                        try:
                            return json.loads(json_str)
                        except Exception:
                            # Try to extract from code block
                            json_str = self.extract_json_from_code_block(response)
                            try:
                                if json_str:
                                    start_index = json_str.find('{')
                                    end_index = json_str.rfind('}')
                                    if start_index != -1 and end_index != -1:
                                        json_str = json_str[start_index:end_index+1]
                                        # Try with escape fixing
                                        fixed_json = self.fix_json_escaping(json_str)
                                        return json.loads(fixed_json)
                            except Exception as e:
                                logger.error(f"Error while parsing json from code block: {str(e)}")
                return {}
            except Exception as e:
                logger.error(f"Error while parsing json: {str(e)}")
                return {}

    def extract_json_from_code_block(self, response: str) -> Optional[str]:
        """
        Extract JSON from a markdown code block.
        
        Args:
            response: Raw response string that might contain markdown code blocks
            
        Returns:
            Extracted JSON string or None if not found
        """
        try:
            pattern = r"```json(.*?)```"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
                return json_content
            return None
        except Exception as e:
            logger.error(f"Error extracting json from code block: {str(e)}")
            return None

