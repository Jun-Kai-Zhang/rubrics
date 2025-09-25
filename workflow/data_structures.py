"""Data structures for the workflow system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Set
import time

@dataclass
class IterationInfo:
    """Information about a single workflow iteration."""
    iteration: int
    responses_per_prompt: Union[int, Dict[str, int], str]
    sample_size: Optional[int]
    start_time: str
    status: str = "in_progress"
    scored_file: Optional[str] = None
    has_ties: bool = False
    prompt_ids_with_ties: List[str] = None
    prompt_ids_without_ties: List[str] = None
    ties_per_prompt: Dict[str, int] = None
    improved_rubrics_file: Optional[str] = None
    next_iteration_responses_file: Optional[str] = None
    grading_mode: Optional[str] = None
    next_responses_per_prompt: Union[int, Dict[str, int], str] = None
    
    def __post_init__(self):
        if self.prompt_ids_with_ties is None:
            self.prompt_ids_with_ties = []
        if self.prompt_ids_without_ties is None:
            self.prompt_ids_without_ties = []
        if self.ties_per_prompt is None:
            self.ties_per_prompt = {}

@dataclass
class TieAnalysis:
    """Results of analyzing scores for ties."""
    has_ties: bool
    tied_responses: List[Dict]
    prompt_ids_with_ties: List[str]
    ties_per_prompt: Dict[str, int]
    prompt_ids_without_ties: List[str]
    responses_without_ties: List[Dict]
    highest_scores_by_prompt: Dict[str, float] = None
    
    def __post_init__(self):
        if self.highest_scores_by_prompt is None:
            self.highest_scores_by_prompt = {}

# WorkflowConfig removed - using Hydra structured configs instead

@dataclass
class WorkflowResult:
    """Final result of workflow execution."""
    status: str
    total_time: float
    iterations: List[IterationInfo]
    final_rubrics_by_prompt: Dict[str, Dict]
    processed_prompt_ids: Optional[Set[str]] = None
    prompt_resolution_tracker: Optional[Dict[str, Dict]] = None
    
    def __post_init__(self):
        if self.processed_prompt_ids is None:
            self.processed_prompt_ids = set()

@dataclass
class MountedWorkflowData:
    """Data loaded from a mounted workflow."""
    processed_prompt_ids: set
    final_rubrics_file: Optional[str]
    metadata: Dict[str, Any] 