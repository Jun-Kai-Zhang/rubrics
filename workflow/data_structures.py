"""Data structures for the workflow system."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union


@dataclass
class IterationInfo:
    """Information about a single workflow iteration."""
    iteration: int
    responses_per_prompt: Union[int, Dict[str, int], str]
    sample_size: Optional[int]
    start_time: str
    status: str = "in_progress"


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
