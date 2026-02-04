from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class AIClient(ABC):
    """
    Abstract base class for AI language model clients.
    """
    
    def __init__(self):
        """Initialize the AI client."""
        pass
    
    @abstractmethod
    async def call_llm(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Make an asynchronous call to a language model.
        
        Args:
            messages: List of message dictionaries in the format expected by the LLM API
            max_tokens: Maximum number of tokens to generate
            model: Optional model name to override the default
            
        Returns:
            Dict containing the response from the language model
        """
        pass
    
    @abstractmethod
    async def parse_json(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from the language model.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed JSON as a dictionary
        """
        pass
