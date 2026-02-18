"""Core business services."""

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


class RubricService:
    """Service for rubric-related operations."""

    def merge_rubrics(
        self,
        current_rubrics: Dict[str, Dict],
        improved_rubrics: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Replace current rubrics with the improved set, unless empty."""
        if not improved_rubrics:
            return current_rubrics.copy()
        return improved_rubrics.copy()

    def filter_rubrics_for_prompts(
        self,
        rubrics: Dict[str, Dict],
        prompt_ids: List[str]
    ) -> Dict[str, Dict]:
        """Filter rubrics to only include specified prompts."""
        return {
            prompt_id: rubrics[prompt_id]
            for prompt_id in prompt_ids
            if prompt_id in rubrics
        }
