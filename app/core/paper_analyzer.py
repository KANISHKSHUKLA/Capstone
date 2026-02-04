import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Any, Optional

from app.clients.gemini_ai import GeminiClient
from app.prompts.templates import PromptTemplates
from app.utils.pdf_processor import PDFProcessor
from app.core.knowledge_graph import KnowledgeGraphExtractor

class PaperAnalyzer:
    """
    Core service for analyzing research papers using parallel LLM calls.
    """
    
    def __init__(self):
        self.llm_client = GeminiClient()
        self.prompt_templates = PromptTemplates()
        self.graph_extractor = KnowledgeGraphExtractor(self.llm_client)
    
    async def save_upload_file(self, file_content: bytes) -> str:
        """
        Save uploaded file content to a temporary location.
        
        Args:
            file_content: Raw bytes of the uploaded file
            
        Returns:
            Path to the saved file
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_path = temp_file.name
                temp_file.write(file_content)
                
            return temp_path
        except Exception as e:
            logging.error(f"Error saving uploaded file: {str(e)}")
            raise
    
    async def analyze_paper(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a research paper by extracting text and making parallel LLM calls.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing the combined results of all analyses
        """
        try:
            paper_text = PDFProcessor.extract_text_from_pdf(file_path)
            title, authors = PDFProcessor.get_paper_metadata(file_path)
            
            if len(paper_text) > 15000:
                paper_text = paper_text[:15000] 
                
            shared_context: Dict[str, Any] = {}
            
            tasks = []
            
            # First call - identify key concepts
            tasks.append(self._analyze_key_concepts(paper_text, shared_context))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            if isinstance(results[0], Dict) and results[0]:
                shared_context["key_concepts"] = results[0]
            
            # Second call - identify problem statement
            problem_task = self._analyze_problem_statement(paper_text, shared_context)
            problem_result = await problem_task
            
            if isinstance(problem_result, Dict) and problem_result:
                shared_context["problem_statement"] = problem_result
            
            # Third call - full explanation
            explanation_task = self._analyze_full_explanation(paper_text, shared_context)
            explanation_result = await explanation_task
            
            if isinstance(explanation_result, Dict) and explanation_result:
                shared_context["full_explanation"] = explanation_result
            
            # Fourth call - generate pseudo code
            pseudocode_task = self._generate_pseudo_code(paper_text, shared_context)
            pseudocode_result = await pseudocode_task
            
            # Fifth call - extract knowledge graph (disabled)
            # knowledge_graph_task = self._extract_knowledge_graph(paper_text, shared_context)
            # knowledge_graph_result = await knowledge_graph_task
            knowledge_graph_result = {"nodes": [], "edges": []}
            
            # Sixth call - in-depth architecture analysis
            architecture_deep_dive_task = self._analyze_architecture_deep_dive(paper_text, shared_context)
            architecture_deep_dive_result = await architecture_deep_dive_task
            
            # Seventh call - generate concrete model.py file code
            model_file_task = self._generate_model_file(paper_text, shared_context)
            model_file_result = await model_file_task
            
            final_result = {
                "metadata": {
                    "title": title or "Unknown Title",
                    "authors": authors or "Unknown Authors",
                },
                "key_concepts": shared_context.get("key_concepts", {}),
                "problem_statement": shared_context.get("problem_statement", {}),
                "full_explanation": shared_context.get("full_explanation", {}),
                "pseudo_code": pseudocode_result or {},
                "knowledge_graph": knowledge_graph_result or {"nodes": [], "edges": []},
                "architecture_deep_dive": architecture_deep_dive_result or {},
                "model_file": model_file_result or ""
            }
            
            return final_result
        except Exception as e:
            logging.error(f"Error analyzing paper: {str(e)}")
            raise
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logging.error(f"Error cleaning up temporary file: {str(e)}")
    
    async def _analyze_key_concepts(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM call to identify key concepts in the paper.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls (empty for first call)
            
        Returns:
            Dictionary containing key concepts analysis
        """
        messages = PromptTemplates.key_concepts_prompt(paper_text)
        return await self.llm_client.call_llm(messages)
    
    async def _analyze_problem_statement(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM call to identify the problem statement and existing approaches.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls
            
        Returns:
            Dictionary containing problem statement analysis
        """
        messages = PromptTemplates.problem_statement_prompt(paper_text, shared_context)
        return await self.llm_client.call_llm(messages)
    
    async def _analyze_full_explanation(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM call to get a comprehensive explanation of the paper.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls
            
        Returns:
            Dictionary containing full explanation
        """
        messages = PromptTemplates.full_explanation_prompt(paper_text, shared_context)
        return await self.llm_client.call_llm(messages)
    
    async def _generate_pseudo_code(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM call to generate pseudo-code implementation based on the paper.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls
            
        Returns:
            Dictionary containing pseudo-code implementation
        """
        messages = PromptTemplates.pseudo_code_prompt(paper_text, shared_context)
        return await self.llm_client.call_llm(messages)
        
    async def _extract_knowledge_graph(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a knowledge graph representing key concepts and their relationships.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls
            
        Returns:
            Dictionary containing nodes and edges for the knowledge graph
        """
        try:
            # Prepare a complete analysis result by combining shared context elements
            analysis_result = {
                "concepts": shared_context.get("key_concepts", {}),
                "problem": shared_context.get("problem_statement", {}),
                "explanation": shared_context.get("full_explanation", {})
            }
            
            # Extract knowledge graph data
            return await self.graph_extractor.extract_graph_data(paper_text, analysis_result)
        except Exception as e:
            logging.error(f"Error extracting knowledge graph: {str(e)}")
            return {"nodes": [], "edges": []}
    
    async def _analyze_architecture_deep_dive(self, paper_text: str, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make LLM call to generate an extremely detailed, in-depth analysis of the architecture.
        This picks apart every mathematical detail, dimension, and design decision.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from other calls including full_explanation
            
        Returns:
            Dictionary containing detailed architecture breakdown
        """
        try:
            messages = PromptTemplates.architecture_deep_dive_prompt(paper_text, shared_context)
            return await self.llm_client.call_llm(messages)
        except Exception as e:
            logging.error(f"Error analyzing architecture deep dive: {str(e)}")
            return {}

    async def _generate_model_file(self, paper_text: str, shared_context: Dict[str, Any]) -> str:
        """
        Make LLM call to generate a complete model.py file in Python representing the architecture.
        The code must include explicit tensor dimensions in comments at each step.
        
        Args:
            paper_text: The text content of the paper
            shared_context: Shared context from prior calls including pseudo_code and architecture_deep_dive
            
        Returns:
            String containing Python code for model.py
        """
        try:
            # Get the architecture deep dive if available, otherwise use full_explanation
            arch_context = shared_context.get("architecture_deep_dive")
            if not arch_context:
                arch_context = shared_context.get("full_explanation", {})
            
            messages = PromptTemplates.model_file_prompt(paper_text, {
                "pseudo_code": shared_context.get("pseudo_code", {}),
                "architecture_deep_dive": arch_context
            })
            
            result = await self.llm_client.call_llm(messages)
            
            # Extract code from the result
            if isinstance(result, dict):
                code = result.get("code", "")
                if code:
                    return code
                # Try alternative keys
                code = result.get("model_py", "") or result.get("model_code", "")
                return code
            elif isinstance(result, str):
                return result
            else:
                logging.warning(f"Unexpected result type for model file: {type(result)}")
                return ""
        except Exception as e:
            logging.error(f"Error generating model file: {str(e)}")
            return ""
