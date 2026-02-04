from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class PaperAnalysisResponse(BaseModel):
    """Response model for paper analysis results."""
    
    class Metadata(BaseModel):
        title: str
        authors: str
    
    class KeyConcept(BaseModel):
        name: str
        category: str
        explanation: str
        relevance: str
    
    class KeyConcepts(BaseModel):
        key_concepts: Optional[List['PaperAnalysisResponse.KeyConcept']] = None
        core_technologies: Optional[List[str]] = None
        novelty_aspects: Optional[List[str]] = None
        field_of_study: Optional[str] = None
        interdisciplinary_connections: Optional[List[str]] = None
    
    class ExistingApproach(BaseModel):
        name: str
        limitations: List[str]
    
    class ProblemStatement(BaseModel):
        problem: Optional[str] = None
        research_questions: Optional[List[str]] = None
        existing_approaches: Optional[List['PaperAnalysisResponse.ExistingApproach']] = None
        gap_in_research: Optional[str] = None
        importance: Optional[str] = None
    
    class Evaluation(BaseModel):
        metrics: Optional[List[str]] = None
        datasets: Optional[List[str]] = None
        baselines: Optional[List[str]] = None
    
    class FullExplanation(BaseModel):
        title: Optional[str] = None
        authors: Optional[str] = None
        approach_summary: Optional[str] = None
        methodology: Optional[str] = None
        innovations: Optional[List[str]] = None
        architecture: Optional[str] = None
        evaluation: Optional['PaperAnalysisResponse.Evaluation'] = None
        results: Optional[str] = None
        limitations: Optional[List[str]] = None
        future_work: Optional[List[str]] = None
    
    class PseudoCodeComponent(BaseModel):
        component: str
        description: str
        code: str
    
    class PseudoCode(BaseModel):
        implementation_overview: Optional[str] = None
        prerequisites: Optional[List[str]] = None
        main_components: Optional[List[str]] = None
        pseudo_code: Optional[List['PaperAnalysisResponse.PseudoCodeComponent']] = None
        usage_example: Optional[str] = None
        potential_challenges: Optional[List[str]] = None
    
    metadata: Metadata
    key_concepts: Optional[KeyConcepts] = Field(default_factory=KeyConcepts)
    problem_statement: Optional[ProblemStatement] = Field(default_factory=ProblemStatement)
    full_explanation: Optional[FullExplanation] = Field(default_factory=FullExplanation)
    pseudo_code: Optional[PseudoCode] = Field(default_factory=PseudoCode)

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    details: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str = Field(..., description="The user's message or question about the paper")
