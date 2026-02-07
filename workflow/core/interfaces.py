"""Domain interfaces and protocols."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol
from workflow.data_structures import TieAnalysis, WorkflowResult


class RubricGenerator(Protocol):
    """Interface for rubric generation."""
    
    def generate_initial_rubrics(
        self, 
        responses: List[Dict],
        model: str
    ) -> Dict[str, Dict]:
        """Generate initial rubrics from responses."""
        ...
    
    def improve_rubrics(
        self,
        scored_responses: Dict,
        current_rubrics: Dict[str, Dict],
        model: str,
        selection_strategy: str = "top2"
    ) -> Dict[str, Dict]:
        """Improve rubrics based on scoring results.

        Args:
            scored_responses: Scored response data
            current_rubrics: Current rubrics by prompt_id
            model: Model name to use for improvement
            selection_strategy: Strategy for selecting reference responses

        Returns:
            Dictionary mapping prompt_id to improved rubric
        """
        ...


class ResponseScorer(Protocol):
    """Interface for response scoring."""
    
    def score_responses(
        self,
        responses: List[Dict],
        rubrics: Dict[str, Dict],
        model: str
    ) -> Dict:
        """Score responses using rubrics."""
        ...


class TieAnalyzer(Protocol):
    """Interface for tie analysis."""
    
    def analyze_ties(
        self,
        scored_data: Dict,
        prompt_ids: List[str]
    ) -> TieAnalysis:
        """Analyze scored data for ties."""
        ...


class WorkflowOrchestrator(ABC):
    """Abstract base class for workflow orchestration."""
    
    @abstractmethod
    def run(self) -> WorkflowResult:
        """Run the workflow."""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Setup workflow dependencies."""
        pass 
