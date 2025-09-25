"""Core business services."""

import logging
from typing import Dict, List, Optional, Set, Tuple
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
        """Merge improved rubrics with current ones."""
        merged = current_rubrics.copy()
        merged.update(improved_rubrics)
        return merged
    
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


class ScoringService:
    """Service for scoring-related operations."""
    
    def extract_top_responses(
        self,
        scored_data: Dict,
        responses_per_prompt: Dict[str, int]
    ) -> List[Dict]:
        """Extract top-scoring responses based on per-prompt limits."""
        extracted_responses = []
        
        for result in scored_data.get("results", []):
            prompt_id = result["id"]
            scored_responses = result["scored_responses"]
            
            if not scored_responses:
                continue
            
            # Sort by score descending
            sorted_responses = sorted(
                scored_responses,
                key=lambda x: x["score"],
                reverse=True
            )
            
            # Take specified number of responses
            num_to_take = responses_per_prompt.get(prompt_id, 1)
            top_responses = sorted_responses[:num_to_take]
            
            extracted_responses.append({
                "id": prompt_id,
                "prompt": result.get("prompt"),
                "responses": [r["response"] for r in top_responses]
            })
        
        return extracted_responses
    
    def calculate_next_iteration_responses(
        self,
        ties_per_prompt: Dict[str, int],
        current_responses: int,
        min_responses: int
    ) -> Dict[str, int]:
        """Calculate responses needed for next iteration."""
        quarter_current = max(1, current_responses // 4)
        
        return {
            prompt_id: max(min_responses, quarter_current, num_tied)
            for prompt_id, num_tied in ties_per_prompt.items()
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
    
    def should_continue_iteration(
        self,
        tie_analysis: TieAnalysis,
        skip_scoring: bool = False
    ) -> Tuple[bool, str]:
        """Determine if workflow should continue to next iteration.
        
        Args:
            tie_analysis: Analysis of ties in current iteration
            skip_scoring: If True, continue on all prompts until max iterations
        
        Returns:
            Tuple of (should_continue, reason)
        """
        # Always continue (force_continue is always True)
        # (max iterations is handled by the loop in workflow_coordinator)
        return True, "force_continue_mode" 