"""Manager for individual workflow iterations."""

import logging
import time
from typing import Dict, List, Tuple

from workflow.data_structures import IterationInfo, TieAnalysis
from workflow.core.services import RubricService, TieAnalysisService
from workflow.core.interfaces import RubricGenerator, ResponseScorer
from workflow.infrastructure import FileHandler

log = logging.getLogger(__name__)


class IterationManager:
    """Manages the execution of a single workflow iteration."""
    
    def __init__(
        self,
        rubric_generator: RubricGenerator,
        response_scorer: ResponseScorer,
        rubric_service: RubricService,
        tie_analysis_service: TieAnalysisService,
        file_handler: FileHandler,
        output_dir: str
    ):
        self.rubric_generator = rubric_generator
        self.response_scorer = response_scorer
        self.rubric_service = rubric_service
        self.tie_analysis_service = tie_analysis_service
        self.file_handler = file_handler
        self.output_dir = output_dir
    
    def run_iteration(
        self,
        iteration_num: int,
        responses: Dict[str, Dict],
        current_rubrics: Dict[str, Dict],
        prompt_ids: List[str],
        config: Dict
    ) -> Tuple[IterationInfo, TieAnalysis, Dict[str, Dict], Dict[str, Dict]]:
        """Run a single iteration of the workflow.
        
        Args:
            iteration_num: Current iteration number
            responses: Response data by prompt_id
            current_rubrics: Current rubrics by prompt_id
            prompt_ids: List of prompt IDs to process
            config: Configuration settings
            
        Returns:
            Tuple of (iteration_info, tie_analysis, updated_rubrics, next_responses)
        """
        log.info(f"Running iteration {iteration_num}")
        
        # Create iteration info
        iter_info = IterationInfo(
            iteration=iteration_num,
            responses_per_prompt=config.get('responses_per_prompt', 64),
            sample_size=len(prompt_ids),
            start_time=time.strftime("%Y-%m-%d_%H-%M-%S")
        )
        
        # Prepare all responses for scoring (no capping or sampling)
        responses_list = self._prepare_responses_for_scoring(responses, prompt_ids)

        # Score all responses
        log.info("Scoring responses...")
        scored_data = self.response_scorer.score_responses(
            responses=responses_list,
            rubrics=current_rubrics,
            model=config['verifier_model']
        )

        # Save scored responses for this iteration
        scored_responses_file = f"{self.output_dir}/iteration_{iteration_num:02d}_scored_responses.json"
        self.file_handler.save_json({
            "metadata": {
                "iteration": iteration_num,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "verifier_model": config['verifier_model'],
                "total_prompts": len(scored_data)
            },
            "scored_responses": scored_data
        }, scored_responses_file)
        log.info(f"Saved scored responses to {scored_responses_file}")

        # Analyze ties
        log.info("Analyzing ties...")
        tie_analysis = self.tie_analysis_service.analyze_scored_responses(
            scored_data=scored_data,
            prompt_ids_to_analyze=set(prompt_ids)
        )
        
        # Update iteration info with tie analysis results
        iter_info.has_ties = tie_analysis.has_ties
        iter_info.prompt_ids_with_ties = tie_analysis.prompt_ids_with_ties
        iter_info.prompt_ids_without_ties = tie_analysis.prompt_ids_without_ties
        iter_info.ties_per_prompt = tie_analysis.ties_per_prompt
        
        # Log detailed resolution status
        if tie_analysis.prompt_ids_without_ties:
            log.info(f"📊 Prompts resolved (no ties) in this iteration: {len(tie_analysis.prompt_ids_without_ties)}")
            for prompt_id in tie_analysis.prompt_ids_without_ties[:5]:  # Show first 5
                score = tie_analysis.highest_scores_by_prompt.get(prompt_id, 'N/A')
                log.info(f"   ✓ {prompt_id}: highest score = {score}")
            if len(tie_analysis.prompt_ids_without_ties) > 5:
                log.info(f"   ... and {len(tie_analysis.prompt_ids_without_ties) - 5} more")
        
        if tie_analysis.prompt_ids_with_ties:
            log.info(f"📊 Prompts with ties: {len(tie_analysis.prompt_ids_with_ties)}")
            for prompt_id in tie_analysis.prompt_ids_with_ties[:5]:  # Show first 5
                num_ties = tie_analysis.ties_per_prompt.get(prompt_id, 0)
                score = tie_analysis.highest_scores_by_prompt.get(prompt_id, 'N/A')
                log.info(f"   ⚡ {prompt_id}: {num_ties} responses tied at score {score}")
            if len(tie_analysis.prompt_ids_with_ties) > 5:
                log.info(f"   ... and {len(tie_analysis.prompt_ids_with_ties) - 5} more")
        
        # Improve rubrics for all prompts using the requested strategy
        improved_rubrics = self.rubric_generator.improve_rubrics(
            scored_responses=scored_data,
            current_rubrics=current_rubrics,
            model=config['rubric_model'],
            selection_strategy=config.get('selection_strategy', 'top2')
        )

        updated_rubrics = self.rubric_service.merge_rubrics(
            current_rubrics, improved_rubrics
        )

        # Save updated rubrics for this iteration
        rubrics_file = f"{self.output_dir}/iteration_{iteration_num:02d}_rubrics.json"
        self.file_handler.save_json({
            "metadata": {
                "iteration": iteration_num,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "rubric_model": config['rubric_model'],
                "total_rubrics": len(updated_rubrics),
                "improved_prompts": len(improved_rubrics),
                "has_ties": tie_analysis.has_ties
            },
            "rubrics": list(updated_rubrics.values())
        }, rubrics_file)
        log.info(f"Saved updated rubrics to {rubrics_file}")

        # Prepare next iteration responses by removing used references
        next_responses = {}
        for prompt_id, prompt_data in responses.items():
            remaining = list(prompt_data.get("responses", []))
            used_info = updated_rubrics.get(prompt_id, {}).get("response_selection_info")
            if config.get('exclude_used_reference_responses', False) and used_info:
                used_texts = {
                    used_info.get("response1_text"),
                    used_info.get("response2_text")
                }
                used_texts.discard(None)
                if used_texts:
                    prior_len = len(remaining)
                    remaining = [resp for resp in remaining if resp not in used_texts]
                    removed = prior_len - len(remaining)
                    if removed:
                        log.info(f"Excluded {removed} used responses for prompt {prompt_id}")
            if len(remaining) >= 2:
                next_responses[prompt_id] = {
                    "id": prompt_id,
                    "prompt": prompt_data.get("prompt", prompt_id),
                    "responses": remaining
                }

        iter_info.status = "completed" if next_responses else "responses_exhausted"
                
        return iter_info, tie_analysis, updated_rubrics, next_responses
    
    def _prepare_responses_for_scoring(
        self,
        responses: Dict[str, Dict],
        prompt_ids: List[str]
    ) -> List[Dict]:
        """Prepare responses in format expected by scorer.
        
        Args:
            responses: Response data by prompt_id
            prompt_ids: List of prompt IDs to process
            
        Returns:
            List of response data formatted for scoring
        """
        formatted_responses = []
        
        for prompt_id in prompt_ids:
            if prompt_id not in responses:
                log.warning(f"No responses found for prompt {prompt_id}")
                continue
            
            response_data = responses[prompt_id]
            
            # Handle different response formats
            if isinstance(response_data, dict):
                if "responses" in response_data:
                    # Standard format
                    formatted_responses.append({
                        "id": prompt_id,
                        "prompt": response_data.get("prompt", prompt_id),
                        "responses": response_data["responses"]
                    })
                else:
                    # Try to extract response list
                    log.warning(f"Unexpected response format for {prompt_id}")
            
        return formatted_responses 
