"""Core business services."""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from workflow.data_structures import TieAnalysis

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


class TieAnalysisService:
    """Service for tie analysis operations."""
    
    def analyze_scored_responses(
        self,
        scored_data: Dict,
        prompt_ids_to_analyze: Set[str]
    ) -> TieAnalysis:
        """Analyze scored responses for ties."""
        has_ties = False
        prompt_ids_with_ties = []
        prompt_ids_without_ties = []
        ties_per_prompt = {}
        tied_responses = []
        responses_without_ties = []
        highest_scores_by_prompt = {}
        
        for result in scored_data.get("results", []):
            prompt_id = result["id"]
            
            if prompt_id not in prompt_ids_to_analyze:
                continue
            
            scored_responses = result["scored_responses"]
            if not scored_responses:
                continue
            
            # Find max score
            max_score = max(resp["score"] for resp in scored_responses)
            highest_scores_by_prompt[prompt_id] = max_score
            
            # Find all responses with max score
            max_score_responses = [
                resp for resp in scored_responses
                if resp["score"] == max_score
            ]
            
            if len(max_score_responses) > 1:
                # Has ties
                has_ties = True
                prompt_ids_with_ties.append(prompt_id)
                ties_per_prompt[prompt_id] = len(max_score_responses)
                
                for resp in max_score_responses:
                    tied_responses.append({
                        "prompt_id": prompt_id,
                        "response": resp.get("response", ""),
                        "response_idx": resp.get("response_idx", 0),
                        "score": resp.get("score", 0)
                    })
            else:
                # No ties
                prompt_ids_without_ties.append(prompt_id)
                if max_score_responses:
                    best_resp = max_score_responses[0]
                    responses_without_ties.append({
                        "prompt_id": prompt_id,
                        "response": best_resp.get("response", ""),
                        "response_idx": best_resp.get("response_idx", 0),
                        "score": best_resp.get("score", 0)
                    })
        
        log.info(
            f"Tie analysis complete: {len(prompt_ids_with_ties)} with ties, "
            f"{len(prompt_ids_without_ties)} without ties"
        )
        
        return TieAnalysis(
            has_ties=has_ties,
            tied_responses=tied_responses,
            prompt_ids_with_ties=prompt_ids_with_ties,
            ties_per_prompt=ties_per_prompt,
            prompt_ids_without_ties=prompt_ids_without_ties,
            responses_without_ties=responses_without_ties,
            highest_scores_by_prompt=highest_scores_by_prompt
        )
    
