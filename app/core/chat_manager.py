import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from app.clients.base import AIClient
from app.prompts.templates import PromptTemplates


class ChatManager:
    """
    Manages chat interactions for research paper analysis.
    This class handles chat history, context management, and LLM interactions.
    """
    
    def __init__(self, llm_client: AIClient):
        """
        Initialize the ChatManager with an LLM client.
        
        Args:
            llm_client: An instance of AIClient to use for LLM interactions
        """
        self.llm_client = llm_client
        self.prompt_templates = PromptTemplates()
    
    async def process_chat_message(
        self, 
        job_id: str,
        user_message: str,
        analysis_result: Dict[str, Any],
        chat_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user's chat message in the context of paper analysis.
        
        Args:
            job_id: Unique identifier for the analysis job
            user_message: The user's chat message/query
            analysis_result: The paper analysis result from the cache
            chat_history: Previous chat interactions, if any
        
        Returns:
            Dictionary containing the processed response and updated chat history
        """
        try:
            if chat_history is None:
                chat_history = []
                
            messages = self._create_chat_messages(user_message, analysis_result, chat_history)
            response = await self.llm_client.call_llm(messages)
            formatted_response = self._format_chat_response(response)
            timestamp = datetime.now().isoformat()
            new_entry = {
                "timestamp": timestamp,
                "query": user_message,
                "response": formatted_response
            }
            
            updated_history = chat_history + [new_entry]
            
            return {
                "response": formatted_response,
                "updated_history": updated_history
            }
        except Exception as e:
            logging.error(f"Error processing chat message: {str(e)}")
            raise
    
    def _create_chat_messages(
        self, 
        user_message: str, 
        analysis_result: Dict[str, Any],
        chat_history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Create chat messages for the LLM using the paper analysis and chat history.
        
        Args:
            user_message: The user's chat message/query
            analysis_result: The paper analysis result from the cache
            chat_history: Previous chat interactions
        
        Returns:
            List of message dictionaries for the LLM
        """
        return self.prompt_templates.chat_prompt(user_message, analysis_result, chat_history)
    
    def _format_chat_response(self, response: Dict[str, Any]) -> str:
        """
        Format the LLM response for display.
        
        Args:
            response: Raw response from the LLM
        
        Returns:
            Formatted response string
        """
        if not response:
            return "I'm sorry, I couldn't generate a response. Please try again."
        
        if isinstance(response, dict) and "answer" in response:
            return response["answer"]
            
        if isinstance(response, dict) and "response" in response:
            return response["response"]
            
        if isinstance(response, dict):
            try:
                return json.dumps(response, indent=2)
            except Exception:
                pass
        return str(response)
