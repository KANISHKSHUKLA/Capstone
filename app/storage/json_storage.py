import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class JSONStorage:
    """
    Handles persistent storage of paper analysis and chat data using JSON files.
    """
    
    def __init__(self):
        try:
            self.storage_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "storage_data")
            self.papers_dir = os.path.join(self.storage_dir, "papers")
            self.chats_dir = os.path.join(self.storage_dir, "chats")
            
            # Create directories if they don't exist
            os.makedirs(self.papers_dir, exist_ok=True)
            os.makedirs(self.chats_dir, exist_ok=True)
            
            logger.info(f"JSON Storage initialized at: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize JSON storage: {str(e)}")
            raise
    
    def _get_paper_path(self, job_id: str) -> str:
        """
        Get the file path for a paper analysis JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            Full path to the paper JSON file
        """
        try:
            return os.path.join(self.papers_dir, f"{job_id}.json")
        except Exception as e:
            logger.error(f"Error getting paper path: {str(e)}")
            raise
    
    def _get_chat_path(self, job_id: str) -> str:
        """
        Get the file path for a chat history JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            Full path to the chat JSON file
        """
        try:
            return os.path.join(self.chats_dir, f"{job_id}.json")
        except Exception as e:
            logger.error(f"Error getting chat path: {str(e)}")
            raise
    
    def save_paper_analysis(self, job_id: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Save paper analysis results to a JSON file.
        
        Args:
            job_id: The unique job identifier
            analysis_data: The complete analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_paper_path(job_id)
            
            # Add metadata
            storage_data = {
                "job_id": job_id,
                "created_at": datetime.now().isoformat(),
                "status": "completed",
                "filename": analysis_data.get("filename", "Unknown"),
                "result": analysis_data.get("result", {})
            }
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(storage_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Successfully saved paper analysis for job: {job_id}")
                return True
            except Exception as e:
                logger.error(f"Error writing paper analysis file: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving paper analysis: {str(e)}")
            return False
    
    def load_paper_analysis(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Load paper analysis results from a JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            Analysis data dictionary if found, None otherwise
        """
        try:
            file_path = self._get_paper_path(job_id)
            
            if not os.path.exists(file_path):
                logger.debug(f"No saved analysis found for job: {job_id}")
                return None
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"Successfully loaded paper analysis for job: {job_id}")
                return data
            except Exception as e:
                logger.error(f"Error reading paper analysis file: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading paper analysis: {str(e)}")
            return None
    
    def save_chat_history(self, job_id: str, chat_history: list) -> bool:
        """
        Save chat history to a JSON file.
        
        Args:
            job_id: The unique job identifier
            chat_history: List of chat messages
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_chat_path(job_id)
            
            chat_data = {
                "job_id": job_id,
                "updated_at": datetime.now().isoformat(),
                "chat_history": chat_history
            }
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chat_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Successfully saved chat history for job: {job_id}")
                return True
            except Exception as e:
                logger.error(f"Error writing chat history file: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False
    
    def load_chat_history(self, job_id: str) -> list:
        """
        Load chat history from a JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            List of chat messages if found, empty list otherwise
        """
        try:
            file_path = self._get_chat_path(job_id)
            
            if not os.path.exists(file_path):
                logger.debug(f"No saved chat history found for job: {job_id}")
                return []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                logger.info(f"Successfully loaded chat history for job: {job_id}")
                return data.get("chat_history", [])
            except Exception as e:
                logger.error(f"Error reading chat history file: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            return []
    
    def delete_paper_analysis(self, job_id: str) -> bool:
        """
        Delete paper analysis JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_paper_path(job_id)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully deleted paper analysis for job: {job_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting paper analysis file: {str(e)}")
                    return False
            else:
                logger.debug(f"No paper analysis file to delete for job: {job_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting paper analysis: {str(e)}")
            return False
    
    def delete_chat_history(self, job_id: str) -> bool:
        """
        Delete chat history JSON file.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = self._get_chat_path(job_id)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully deleted chat history for job: {job_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting chat history file: {str(e)}")
                    return False
            else:
                logger.debug(f"No chat history file to delete for job: {job_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting chat history: {str(e)}")
            return False
    
    def paper_exists(self, job_id: str) -> bool:
        """
        Check if a paper analysis exists.
        
        Args:
            job_id: The unique job identifier
            
        Returns:
            True if exists, False otherwise
        """
        try:
            file_path = self._get_paper_path(job_id)
            return os.path.exists(file_path)
        except Exception as e:
            logger.error(f"Error checking paper existence: {str(e)}")
            return False
    
    def list_all_papers(self) -> list:
        """
        List all stored paper analysis job IDs.
        
        Returns:
            List of job IDs
        """
        try:
            try:
                files = os.listdir(self.papers_dir)
                job_ids = [f.replace('.json', '') for f in files if f.endswith('.json')]
                return job_ids
            except Exception as e:
                logger.error(f"Error listing papers: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Error in list_all_papers: {str(e)}")
            return []

