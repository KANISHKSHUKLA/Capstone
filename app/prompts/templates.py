from typing import Dict, List, Any, Optional

class PromptTemplates:
    """
    Contains prompt templates for different aspects of research paper analysis.
    """
    
    @staticmethod
    def key_concepts_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt to identify key concepts in a research paper.
        
        Args:
            paper_content: The text content of the research paper
            other_contexts: Optional context from other parallel calls
            
        Returns:
            List of message dictionaries for the LLM
        """
        context_info = ""
        if other_contexts:
            context_info = "\nAdditional context from parallel analysis:\n"
            if "problem_statement" in other_contexts:
                context_info += f"- Problem being addressed: {other_contexts['problem_statement'].get('problem', 'Not yet available')}\n"
            if "implementation" in other_contexts:
                context_info += f"- Approach overview: {other_contexts['implementation'].get('approach_summary', 'Not yet available')}\n"
        
        return [
            {"role": "system", "content": """You are an expert academic researcher specializing in analyzing research papers. 
Your task is to identify and explain the key concepts, technologies, and methodologies used in a research paper.
Provide your response in JSON format."""},
            {"role": "user", "content": f"""Analyze the following research paper and identify the key concepts, technologies, frameworks, and methodologies used.
For each concept, provide a brief explanation of what it is and how it's used in the paper.
{context_info}

Research Paper Content:
{paper_content[:15000]}  # Limit paper content to avoid token limit issues

Format your response as a JSON object with the following structure:
{{
  "key_concepts": [
    {{
      "name": "concept name",
      "category": "algorithm/framework/methodology/technology/etc",
      "explanation": "explanation of the concept",
      "relevance": "how this concept is used in the paper"
    }}
  ],
  "core_technologies": ["list of main technologies used"],
  "novelty_aspects": ["list of novel approaches introduced in the paper"],
  "field_of_study": "the primary academic field this research belongs to",
  "interdisciplinary_connections": ["fields or domains this research connects"]
}}"""}
        ]

    @staticmethod
    def problem_statement_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt to identify the problem statement and challenges in a research paper.
        
        Args:
            paper_content: The text content of the research paper
            other_contexts: Optional context from other parallel calls
            
        Returns:
            List of message dictionaries for the LLM
        """
        context_info = ""
        if other_contexts:
            context_info = "\nAdditional context from parallel analysis:\n"
            if "key_concepts" in other_contexts:
                key_concepts = other_contexts["key_concepts"].get("core_technologies", ["Not yet available"])
                context_info += f"- Core technologies identified: {', '.join(key_concepts[:3] if isinstance(key_concepts, list) else [key_concepts])}\n"
            if "full_explanation" in other_contexts:
                approach = other_contexts["full_explanation"].get("approach_summary", "Not yet available")
                context_info += f"- Approach summary: {approach}\n"
        
        return [
            {"role": "system", "content": """You are an expert academic researcher specializing in analyzing research papers.
Your task is to identify and clearly articulate the problem statement, challenges, and limitations of existing approaches addressed in a research paper.
Provide your response in JSON format."""},
            {"role": "user", "content": f"""Analyze the following research paper and identify:
1. The main problem statement or research question being addressed
2. What alternative approaches or methods exist for solving this problem
3. The limitations, challenges, or blockers in these existing methods
4. Why a new approach was necessary
{context_info}

Research Paper Content:
{paper_content[:15000]}  # Limit paper content to avoid token limit issues

Format your response as a JSON object with the following structure:
{{
  "problem": "concise statement of the core problem being addressed",
  "research_questions": ["list of specific research questions"],
  "existing_approaches": [
    {{
      "name": "name or description of existing approach",
      "limitations": ["specific limitations or challenges of this approach"]
    }}
  ],
  "gap_in_research": "explanation of what was missing in existing approaches",
  "importance": "why solving this problem is significant to the field"
}}"""}
        ]

    @staticmethod
    def full_explanation_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt for comprehensive end-to-end explanation of the research paper.
        
        Args:
            paper_content: The text content of the research paper
            other_contexts: Optional context from other parallel calls
            
        Returns:
            List of message dictionaries for the LLM
        """
        context_info = ""
        if other_contexts:
            context_info = "\nAdditional context from parallel analysis:\n"
            if "key_concepts" in other_contexts:
                field = other_contexts["key_concepts"].get("field_of_study", "Not yet available")
                context_info += f"- Field of study: {field}\n"
            if "problem_statement" in other_contexts:
                problem = other_contexts["problem_statement"].get("problem", "Not yet available")
                context_info += f"- Problem being addressed: {problem}\n"
        
        return [
            {"role": "system", "content": """You are an expert academic researcher specializing in analyzing research papers.
Your task is to provide a comprehensive explanation of a research paper, covering its approach, methodology, innovations, evaluation metrics, and results.
Provide your response in JSON format."""},
            {"role": "user", "content": f"""Analyze the following research paper and provide a comprehensive explanation covering:
1. The overall approach and methodology
2. The key innovations or novel contributions
3. The architecture or system design
4. The evaluation metrics used
5. The main results and their implications
6. The limitations acknowledged by the authors
7. Future work suggested
{context_info}

Research Paper Content:
{paper_content[:15000]}  # Limit paper content to avoid token limit issues

Format your response as a JSON object with the following structure:
{{
  "title": "inferred title of the paper",
  "authors": "inferred authors if mentioned",
  "approach_summary": "concise summary of the approach taken",
  "methodology": "detailed explanation of the methodology",
  "innovations": ["list of key novel contributions"],
  "architecture": "description of the system architecture or design",
  "evaluation": {{
    "metrics": ["list of evaluation metrics used"],
    "datasets": ["datasets used if applicable"],
    "baselines": ["baseline methods compared against"]
  }},
  "results": "summary of main results and performance",
  "limitations": ["limitations acknowledged in the paper"],
  "future_work": ["directions for future work mentioned"]
}}"""}
        ]

    @staticmethod
    def pseudo_code_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt to generate pseudo-code implementation based on the research paper.
        
        Args:
            paper_content: The text content of the research paper
            other_contexts: Optional context from other parallel calls
            
        Returns:
            List of message dictionaries for the LLM
        """
        context_info = ""
        if other_contexts:
            context_info = "\nAdditional context from parallel analysis:\n"
            if "key_concepts" in other_contexts:
                technologies = other_contexts["key_concepts"].get("core_technologies", ["Not yet available"])
                context_info += f"- Core technologies: {', '.join(technologies[:3] if isinstance(technologies, list) else [technologies])}\n"
            if "full_explanation" in other_contexts:
                methodology = other_contexts["full_explanation"].get("methodology", "Not yet available")
                architecture = other_contexts["full_explanation"].get("architecture", "Not yet available")
                context_info += f"- Methodology: {methodology[:200]}...\n"
                context_info += f"- Architecture: {architecture[:200]}...\n"
        
        return [
            {"role": "system", "content": """You are an expert AI and machine learning engineer specializing in implementing research papers.
Your task is to generate clear, well-structured pseudo-code that implements the core algorithms and methods described in a research paper.
Provide your response in JSON format."""},
            {"role": "user", "content": f"""Based on the following research paper, generate well-structured pseudo-code that implements the core algorithms, methods, or architecture described in the paper.
Focus on the most important and novel aspects of the paper implementation.
Provide code that is clear, commented, and follows best practices.
{context_info}

Research Paper Content:
{paper_content[:15000]}  # Limit paper content to avoid token limit issues

Format your response as a JSON object with the following structure:
{{
  "implementation_overview": "brief description of what the code implements",
  "prerequisites": ["libraries, frameworks, or dependencies needed"],
  "main_components": ["list of key components in the implementation"],
  "pseudo_code": [
    {{
      "component": "name of component or algorithm",
      "description": "what this component does",
      "code": "detailed pseudo-code implementation with comments"
    }}
  ],
  "usage_example": "example of how to use the implemented code",
  "potential_challenges": ["implementation challenges to be aware of"]
}}"""}
        ]

    @staticmethod
    def architecture_deep_dive_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt for an extremely detailed, in-depth analysis of the architecture and methodology.
        This prompt picks apart every detail, mathematical formulation, dimension, and design decision.
        
        Args:
            paper_content: The text content of the research paper
            other_contexts: Context from previous analysis including architecture description
            
        Returns:
            List of message dictionaries for the LLM
        """
        architecture = ""
        methodology = ""
        innovations = []
        
        if other_contexts and "full_explanation" in other_contexts:
            full_exp = other_contexts["full_explanation"]
            architecture = full_exp.get("architecture", "")
            methodology = full_exp.get("methodology", "")
            innovations = full_exp.get("innovations", [])
        
        return [
            {"role": "system", "content": """You are an exceptionally detail-oriented academic researcher and AI architect with deep expertise in dissecting complex technical systems.

Your task is to provide an EXTREMELY DETAILED, bone-deep analysis of the paper's architecture and methodology. Go far beyond surface-level descriptions.

For every component, explain:
- The exact mathematical formulations and operations
- The precise dimensions and shapes at each step
- Why specific design decisions were made
- How information flows through the system
- The computational complexity and efficiency considerations
- The intuition behind architectural choices
- How different components interact and affect each other

Think of this as teaching someone who wants to implement this from scratch with complete understanding of every detail.

CRITICAL JSON FORMATTING RULES:
1. Your response MUST be valid JSON
2. ALL backslashes in LaTeX must be DOUBLE-ESCAPED in JSON strings
3. Example: Write "\\\\sigma" not "\\sigma", "\\\\text{}" not "\\text{}"
4. Example: Write "\\\\frac{1}{2}" not "\\frac{1}{2}"
5. Always escape special characters: quotes as \\", newlines as \\n
6. Test your JSON is valid before responding"""},
            {"role": "user", "content": f"""Based on the research paper below, provide an exhaustive, meticulous breakdown of the architecture and methodology.

**Previously Identified Architecture:**
{architecture}

**Previously Identified Methodology:**
{methodology}

**Key Innovations:**
{', '.join(innovations) if innovations else 'See paper content'}

Now, I need you to go MUCH deeper. For each major component of the architecture:

1. **Break down the mathematical operations** - What exact transformations happen? What are the equations?
2. **Explain dimensions at every step** - If data flows through layers, what are the input/output dimensions? How do shapes change?
3. **Detail the internal mechanisms** - For neural network layers, explain what happens internally. For algorithms, break down each step.
4. **Explain design rationale** - WHY was this component designed this way? What alternatives exist and why weren't they chosen?
5. **Trace information flow** - How does data transform as it moves through the system? What information is preserved or lost?
6. **Identify subtle but critical details** - Normalization techniques, activation functions, initialization strategies, gating mechanisms, attention patterns, etc.
7. **Connect to the problem** - How does each architectural choice address the specific challenges identified in the paper?

Research Paper Content:
{paper_content[:15000]}

Format your response as a JSON object with the following structure:
{{
  "overview": "A brief summary of what you'll explain",
  "detailed_breakdown": [
    {{
      "component_name": "Name of the architectural component or stage",
      "purpose": "What this component is designed to achieve",
      "detailed_explanation": "Extremely detailed explanation covering math, dimensions, operations, design rationale. Use $...$ for inline math and $$...$$ for display equations in LaTeX format.",
      "mathematical_formulation": "Key equations or mathematical operations. IMPORTANT: Use proper LaTeX format with $...$ for inline math (e.g., $x^2 + y^2$) and $$...$$ for display equations. Variables like vectors/matrices should be in LaTeX, e.g., $W_{{conv}}$, $\\\\\\\\sigma(x)$, $h_{{k,j}}$. CRITICAL: In JSON, ALL backslashes must be DOUBLE-ESCAPED (write \\\\\\\\ for each backslash). Use \\\\\\\\text{{}} for text in equations.",
      "dimension_analysis": "Input/output dimensions, shape transformations, tensor operations. Use LaTeX for mathematical notation.",
      "design_rationale": "Why this design? What problem does it solve? What alternatives were considered?",
      "subtle_details": "Critical but often overlooked implementation details"
    }}
  ],
  "integration_flow": "How all components work together end-to-end, with specific attention to how information flows and transforms. Use LaTeX for any mathematical notation.",
  "critical_insights": ["Key insights about why this architecture works well for the problem"],
  "implementation_considerations": ["Important details for anyone implementing this system"]
}}"""}
        ]

    @staticmethod
    def model_file_prompt(paper_content: str, other_contexts: Dict[str, Any] = None) -> List[Dict[str, str]]:
        """
        Create a prompt that asks for a complete Python model.py representing the architecture.
        The code must include explicit tensor dimensions at each step as comments.
        It must be self-contained and avoid assumptions; simplified is fine but explicit.
        
        Inputs include the pseudo-code and architecture deep dive if available.
        """
        pseudo_code = {}
        deep_dive = {}
        if other_contexts:
            pseudo_code = other_contexts.get("pseudo_code", {}) or {}
            deep_dive = other_contexts.get("architecture_deep_dive", {}) or {}

        return [
            {"role": "system", "content": """You are a meticulous ML engineer who produces clean, self-contained Python implementations.
Your task is to generate a complete model.py file that implements the paper's architecture.
You MUST respond with valid JSON format.

CRITICAL: In JSON strings, escape backslashes by doubling them (e.g., "\\\\n" for newline in Python code)."""},
            {"role": "user", "content": f"""Resources to base the implementation on:

1) Pseudo-code (selected parts):
{str(pseudo_code)[:10000]}

2) Architecture deep dive (selected parts):
{str(deep_dive)[:10000]}

3) Paper excerpt for reference:
{paper_content[:10000]}

Generate a complete Python model.py file that implements the paper's architecture.

Requirements:
- Use PyTorch (torch, torch.nn as nn, torch.nn.functional as F) unless otherwise specified
- No runtime assumptions; expose unknowns as constructor args with docstrings
- At every transformation, include comments with exact tensor dimensions, e.g. `# [batch, seq_len, 512] -> [batch, seq_len, 1024] and then a real example [1, 10, 512] -> [1, 10, 1024]`
- Include a top-level docstring summarizing the model and expected input shapes
- Include a small `if __name__ == "__main__":` smoke test that instantiates the model and runs forward pass
- Do not include training loops; focus on architecture only

Format your response as a JSON object with the following structure:
{{
  "code": "complete Python code for model.py as a string"
}}"""}
        ]

    @staticmethod
    def chat_prompt(user_message: str, analysis_result: Dict[str, Any], chat_history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        """
        Create a prompt for chat interactions about a previously analyzed paper.
        
        Args:
            user_message: The user's query or message
            analysis_result: The complete paper analysis result
            chat_history: Previous chat interactions, if any
            
        Returns:
            List of message dictionaries for the LLM
        """
        system_content = """You are an expert academic researcher and AI assistant specializing in explaining research papers.

You have previously analyzed a research paper and provided detailed information about its key concepts, problem statement, 
methodology, results, and generated pseudo-code based on the paper.

Your task is to answer questions about this paper using the analysis you've already done. Provide clear, 
accurate, and helpful responses based specifically on the paper content and your analysis.

You should respond conversationally but maintain academic rigor. If a question falls outside the scope of the paper 
or your analysis, acknowledge this limitation politely.

Provide your response in a structured format with appropriate markdown formatting when needed."""
        
        paper_context = "Paper Analysis Summary:\n"
        
        if "metadata" in analysis_result:
            metadata = analysis_result["metadata"]
            title = metadata.get("title", "Unknown")
            authors = metadata.get("authors", "Unknown")
            paper_context += f"Title: {title}\nAuthors: {authors}\n\n"
        
        if "key_concepts" in analysis_result:
            key_concepts = analysis_result["key_concepts"]
            if "core_technologies" in key_concepts and key_concepts["core_technologies"]:
                core_tech = ", ".join(key_concepts["core_technologies"] if isinstance(key_concepts["core_technologies"], list) else [key_concepts["core_technologies"]])
                paper_context += f"Core Technologies: {core_tech}\n"
            if "field_of_study" in key_concepts:
                paper_context += f"Field: {key_concepts['field_of_study']}\n"
        
        if "problem_statement" in analysis_result:
            problem = analysis_result["problem_statement"].get("problem", "")
            if problem:
                paper_context += f"Problem: {problem}\n"
        
        if "full_explanation" in analysis_result:
            approach = analysis_result["full_explanation"].get("approach_summary", "")
            if approach:
                paper_context += f"Approach: {approach}\n"
        
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            chat_context = "Previous conversation:\n"
            recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
            for entry in recent_history:
                chat_context += f"User: {entry['query']}\n"
                chat_context += f"Assistant: {entry['response']}\n\n"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{paper_context}\n{chat_context}\nUser's question: {user_message}\n\nFormat your response as a JSON object with the following structure:\n{{\n  \"answer\": \"your comprehensive answer to the user's question\"\n}}"}
        ]
        
        return messages
