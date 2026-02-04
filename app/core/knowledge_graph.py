import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class KnowledgeGraphExtractor:
    """
    Extracts knowledge graph data from research paper analysis results.
    Creates a graph representation with concepts as nodes and their relationships as edges.
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def extract_graph_data(self, paper_text: str, analysis_result: Dict[str, Any]) -> Dict:
        """
        Extract nodes and edges for a knowledge graph from paper text and existing analysis.
        
        Args:
            paper_text: The full text of the research paper
            analysis_result: The existing analysis results containing concepts, problem statement, etc.
            
        Returns:
            A dictionary containing nodes and edges for the knowledge graph
        """
        # Create a specialized prompt for the LLM
        prompt = self._create_graph_extraction_prompt(paper_text, analysis_result)
        
        try:
            # Call the LLM to extract graph structure
            messages = [
                {"role": "system", "content": """You are an expert at extracting conceptual relationships from research papers. 
CRITICAL: You MUST respond with valid JSON only. Escape all special characters properly (use \\n for newlines, \\" for quotes). 
No control characters allowed in string values. Ensure all node IDs are unique strings."""},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm_client.call_llm(messages)
            
            # The response should already be parsed JSON from the client
            # But we'll validate and process it
            if isinstance(response, dict) and "nodes" in response and "edges" in response:
                graph_data = response
            else:
                # If it's not in the expected format, try to process it
                graph_data = self._process_graph_response(response)
            
            # Add graph metadata
            paper_title = analysis_result.get("title", "")
            if not paper_title and "explanation" in analysis_result:
                paper_title = analysis_result["explanation"].get("title", "Untitled Paper")
            
            graph_data["metadata"] = {
                "paper_title": paper_title or "Untitled Paper",
                "node_count": len(graph_data.get("nodes", [])),
                "edge_count": len(graph_data.get("edges", []))
            }
            
            return graph_data
            
        except Exception as e:
            logger.error(f"Error extracting knowledge graph: {str(e)}")
            # Return empty graph structure on error
            return {"nodes": [], "edges": [], "metadata": {"error": str(e)}}
    
    def _create_graph_extraction_prompt(self, paper_text: str, analysis_result: Dict[str, Any]) -> str:
        """Create a prompt for extracting knowledge graph data"""
        # Extract useful information from analysis result
        concepts_data = analysis_result.get("concepts", {})
        
        # Handle key_concepts - could be list of strings or list of dicts
        raw_key_concepts = concepts_data.get("key_concepts", [])
        if isinstance(raw_key_concepts, list):
            key_concepts = []
            for k in raw_key_concepts:
                if isinstance(k, dict):
                    key_concepts.append(k.get("name", str(k)))
                else:
                    key_concepts.append(str(k))
        else:
            key_concepts = []
            
        # Extract core_technologies
        raw_technologies = concepts_data.get("core_technologies", [])
        if isinstance(raw_technologies, list):
            technologies = [str(t) for t in raw_technologies if t is not None]
        else:
            technologies = []
        
        # Extract problem statement
        problem_data = analysis_result.get("problem", {})
        problem_statement = str(problem_data.get("problem", problem_data.get("problem_statement", "")))
        
        # Extract existing approaches
        existing_approaches = problem_data.get("existing_approaches", [])
        approaches = []
        if isinstance(existing_approaches, list):
            for a in existing_approaches:
                if isinstance(a, dict):
                    approaches.append(a.get("name", str(a)))
                else:
                    approaches.append(str(a))
        
        # Extract methodology
        explanation = analysis_result.get("explanation", {})
        methodology = str(explanation.get("methodology", explanation.get("approach_summary", "")))
        
        # Format the concepts and technologies for display
        concepts_str = ", ".join(key_concepts[:10]) if key_concepts else "None identified yet"
        tech_str = ", ".join(technologies[:10]) if technologies else "None identified yet"
        approaches_str = ", ".join(approaches[:5]) if approaches else "None identified yet"
        
        prompt = f"""Extract a detailed and comprehensive knowledge graph from this research paper that reveals the paper's core contributions and how concepts interconnect.

I'll provide you with the paper text and concepts we've already identified. Your task is to create an in-depth graph that captures the paper's intellectual framework.

ALREADY IDENTIFIED INFORMATION:
- Key concepts: {concepts_str}
- Technologies: {tech_str}
- Problem statement: {problem_statement[:300] if problem_statement else "Not yet identified"}
- Alternative approaches: {approaches_str}
- Methodology: {methodology[:300] if methodology else "Not yet identified"}

INSTRUCTIONS:
1. Create 20-35 nodes representing:
   - Core theoretical concepts and their foundations
   - Specific techniques, methods, and algorithms
   - Datasets, benchmarks, and evaluation metrics
   - Results, findings, and implications
   - Limitations, challenges, and future directions
   - Related work and competing approaches

2. Create 30-60 edges showing detailed relationships between nodes

3. Node types (use these exact types):
   - "concept": Theoretical ideas, principles, or frameworks
   - "method": Specific techniques, algorithms, or approaches
   - "dataset": Data sources, benchmarks, or test environments
   - "result": Findings, outcomes, or performance metrics
   - "entity": People, organizations, or systems mentioned
   - "limitation": Constraints, issues, or shortcomings identified
   - "application": Practical uses or implementations

4. Relationship types (use these exact types):
   - "uses": One concept/method utilizes another
   - "improves": Enhances or builds upon previous work
   - "contradicts": Challenges or refutes another concept
   - "part_of": Represents a component relationship
   - "results_in": Produces a specific outcome
   - "depends_on": Requires another concept/method
   - "applied_to": Implemented in a specific context
   - "evaluated_on": Tested against a particular benchmark
   - "addresses": Attempts to solve a specific problem
   - "compared_with": Contrasted against another approach
   - "leads_to": Insights or directions for future work

CRITICAL JSON FORMATTING:
- Escape all special characters: use \\\\n for newlines, \\\\" for quotes
- No control characters allowed in string values
- All node IDs must be unique strings
- All descriptions must be valid JSON strings

OUTPUT FORMAT - Return ONLY valid JSON:
{{
  "nodes": [
    {{"id": "node_1", "label": "Concept Name", "type": "concept", "description": "Brief description"}},
    {{"id": "node_2", "label": "Method Name", "type": "method", "description": "Brief description"}}
  ],
  "edges": [
    {{"source": "node_1", "target": "node_2", "label": "uses", "description": "explanation"}}
  ]
}}

Paper text excerpt:
{paper_text[:5000]}"""
        
        return prompt
    
    def _process_graph_response(self, response) -> Dict:
        """Process and validate the LLM response"""
        try:
            # Check if response is already a dictionary (parsed JSON)
            if isinstance(response, dict):
                graph_data = response
            elif isinstance(response, str):
                # Try to parse as JSON directly
                try:
                    graph_data = json.loads(response)
                except json.JSONDecodeError:
                    # Extract JSON from code blocks or text
                    json_content = response
                    if "```json" in response:
                        json_content = response.split("```json")[1].split("```")[0].strip()
                    elif "```" in response:
                        json_content = response.split("```")[1].split("```")[0].strip()
                    
                    # Try to find JSON boundaries
                    start_idx = json_content.find('{')
                    end_idx = json_content.rfind('}')
                    if start_idx >= 0 and end_idx > start_idx:
                        json_content = json_content[start_idx:end_idx+1]
                    
                    graph_data = json.loads(json_content)
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return {"nodes": [], "edges": []}
            
            # Validate required keys
            if not all(k in graph_data for k in ["nodes", "edges"]):
                raise ValueError("Missing required keys in graph data")
                
            # Ensure all nodes have required fields
            for node in graph_data["nodes"]:
                if not all(k in node for k in ["id", "label", "type"]):
                    node["id"] = node.get("id", f"node_{len(graph_data['nodes'])}")
                    node["label"] = node.get("label", "Unnamed Concept")
                    node["type"] = node.get("type", "concept")
                if "description" not in node:
                    node["description"] = f"A {node['type']} in the paper"
            
            # Validate node references in edges
            node_ids = {node["id"] for node in graph_data["nodes"]}
            valid_edges = []
            
            for edge in graph_data["edges"]:
                if not all(k in edge for k in ["source", "target", "label"]):
                    continue
                
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    if "description" not in edge:
                        edge["description"] = f"A {edge['label']} relationship"
                    valid_edges.append(edge)
            
            graph_data["edges"] = valid_edges
            
            return graph_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"Error processing graph response: {str(e)}")
            return {"nodes": [], "edges": []}
