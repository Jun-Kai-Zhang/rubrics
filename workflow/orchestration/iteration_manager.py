"""Manager for individual workflow iterations."""

import logging
import random
import time
from typing import Dict, List, Optional, Tuple

from workflow.data_structures import IterationInfo, TieAnalysis
from workflow.core.services import RubricService, ScoringService, TieAnalysisService
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
        scoring_service: ScoringService,
        tie_analysis_service: TieAnalysisService,
        file_handler: FileHandler,
        output_dir: str
    ):
        self.rubric_generator = rubric_generator
        self.response_scorer = response_scorer
        self.rubric_service = rubric_service
        self.scoring_service = scoring_service
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
        
        # Limit responses for grading with random sampling to avoid order bias
        # - Always apply on the first iteration if configured
        # - Additionally, when grade_all_responses or skip_scoring is enabled, re-sample each iteration
        working_responses = responses
        target_per_prompt = config.get('responses_per_prompt')
        # Always grade all responses
        if isinstance(target_per_prompt, int) and target_per_prompt > 0 and (
            iteration_num == 1 or True or config.get('skip_scoring', False)
        ):
            capped: Dict[str, Dict] = {}
            for prompt_id in prompt_ids:
                if prompt_id not in responses:
                    continue
                prompt_data = responses[prompt_id]
                # Only cap if the standard format is present
                if isinstance(prompt_data, dict) and 'responses' in prompt_data and isinstance(prompt_data['responses'], list):
                    response_list = prompt_data['responses']
                    if len(response_list) > target_per_prompt:
                        # Randomly sample without replacement
                        sampled = random.sample(response_list, target_per_prompt)
                    else:
                        sampled = response_list
                    new_entry = dict(prompt_data)
                    new_entry['responses'] = sampled
                    capped[prompt_id] = new_entry
                else:
                    capped[prompt_id] = prompt_data
            working_responses = capped

        # Prepare responses for scoring from the possibly capped working set
        responses_list = self._prepare_responses_for_scoring(
            working_responses, prompt_ids
        )
        
        # Skip scoring if skip_scoring is True
        if config.get('skip_scoring', False):
            log.info("Skipping scoring - skip_scoring mode enabled")
            
            # Create a dummy scored_data structure with all responses
            # This is needed for the improve_rubrics function to work
            # By setting all scores to 0, all responses become "tied" at the highest score,
            # so the existing logic in find_prompts_with_tied_highest_scores will
            # randomly select 2 responses from all available responses
            scored_data = {
                "results": []
            }
            
            for prompt_id in prompt_ids:
                if prompt_id in working_responses:
                    prompt_data = working_responses[prompt_id]
                    scored_responses = []
                    
                    # Convert responses to scored format (without actual scores)
                    for idx, response in enumerate(prompt_data.get('responses', [])):
                        scored_responses.append({
                            "response": response,
                            "response_idx": idx,
                            "prompt_id": prompt_id,
                            "score": 0,  # Dummy score
                            "score_text": "Not scored - skip_scoring mode"
                        })
                    
                    scored_data["results"].append({
                        "id": prompt_id,
                        "prompt": prompt_data.get('prompt', ''),
                        "rubric": current_rubrics.get(prompt_id, {}).get('criteria', []),
                        "scored_responses": scored_responses
                    })
            
            # Create dummy tie analysis - all prompts are considered to have "ties"
            tie_analysis = TieAnalysis(
                has_ties=True,
                tied_responses=[],
                prompt_ids_with_ties=prompt_ids,
                ties_per_prompt={pid: 2 for pid in prompt_ids},  # Dummy value
                prompt_ids_without_ties=[],
                responses_without_ties=[],
                highest_scores_by_prompt={pid: 0 for pid in prompt_ids}  # Dummy scores
            )
            
        else:
            # Normal mode: Score responses
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
        
        # Initialize updated rubrics with current rubrics
        updated_rubrics = current_rubrics.copy()
        next_responses = {}
        
        # If there are ties, improve rubrics and prepare next iteration
        if tie_analysis.has_ties:
            log.info(f"Found ties in {len(tie_analysis.prompt_ids_with_ties)} prompts")
            
            # Improve rubrics for tied prompts
            improved_rubrics = self.rubric_generator.improve_rubrics(
                scored_responses=scored_data,
                current_rubrics=current_rubrics,
                model=config['rubric_model']
            )
            
            # Merge improved rubrics
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
                    "has_ties": True
                },
                "rubrics": list(updated_rubrics.values())
            }, rubrics_file)
            log.info(f"Saved updated rubrics to {rubrics_file}")
            
            if True:  # Always keep all responses
                # Keep all responses for all prompts
                log.info("Grade all responses + Force continue mode: keeping all responses for all prompts")
                # Keep the full pool so a fresh sample can be drawn next iteration
                next_responses = responses
            elif config.get('skip_scoring', False):
                # In skip scoring mode, keep all responses for all prompts
                # Keep the full pool so a fresh sample can be drawn next iteration
                next_responses = responses
            else:
                # Calculate responses for next iteration (reduced set)
                responses_per_prompt = self.scoring_service.calculate_next_iteration_responses(
                    ties_per_prompt=tie_analysis.ties_per_prompt,
                    current_responses=config.get('responses_per_prompt', 64),
                    min_responses=4
                )
                
                # Extract top responses for next iteration
                top_responses = self.scoring_service.extract_top_responses(
                    scored_data=scored_data,
                    responses_per_prompt=responses_per_prompt
                )
                
                # Convert to dict format
                next_responses = {
                    resp['id']: resp 
                    for resp in top_responses
                }

                # Always exclude the two responses used for improvement from next iteration
                try:
                    for result in scored_data.get("results", []):
                        pid = result.get("id")
                        if pid not in next_responses:
                            continue
                        # Pull used reference responses from updated rubric metadata if present
                        used_info = updated_rubrics.get(pid, {}).get("response_selection_info")
                        if not used_info:
                            continue
                        r1 = used_info.get("response1_text")
                        r2 = used_info.get("response2_text")
                        if not r1 and not r2:
                            continue
                        resp_list = next_responses[pid].get("responses", [])
                        # Filter by exact text match
                        filtered = [r for r in resp_list if r != r1 and r != r2]
                        if len(filtered) != len(resp_list):
                            log.info(f"Excluded used references for prompt {pid}: removed {len(resp_list) - len(filtered)} responses")
                        next_responses[pid]["responses"] = filtered
                except Exception as _ex:
                    log.warning(f"Failed to exclude used reference responses: {_ex}")
            
            iter_info.status = "completed"
        else:
            # Handle case when skip_scoring is True but no ties found
            if config.get('skip_scoring', False):
                # This should never happen since skip_scoring mode always has "ties"
                log.warning("Unexpected state: skip_scoring is True but no ties found")
                iter_info.status = "error_unexpected_state"
            else:  # Always force_continue
                # Force continue mode: even with no ties at highest score, continue
                log.info("No ties found at highest score, but force_continue is enabled")
                
                # We need to improve rubrics even when there are no ties at highest score
                improved_rubrics = self.rubric_generator.improve_rubrics(
                    scored_responses=scored_data,
                    current_rubrics=current_rubrics,
                    model=config['rubric_model']
                )
                
                # Merge improved rubrics
                updated_rubrics = self.rubric_service.merge_rubrics(
                    current_rubrics, improved_rubrics
                )
                
                # Save updated rubrics for this iteration when force_continue is True
                rubrics_file = f"{self.output_dir}/iteration_{iteration_num:02d}_rubrics.json"
                self.file_handler.save_json({
                    "metadata": {
                        "iteration": iteration_num,
                        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "rubric_model": config['rubric_model'],
                        "total_rubrics": len(updated_rubrics),
                        "improved_prompts": len(improved_rubrics),
                        "has_ties": False,
                        "status": "force_continue_no_ties"
                    },
                    "rubrics": list(updated_rubrics.values())
                }, rubrics_file)
                log.info(f"Saved updated rubrics (force_continue) to {rubrics_file}")
                
                # Keep all responses for all prompts
                log.info("Force continue mode: keeping all responses for all prompts")
                next_responses = responses
                
                iter_info.status = "completed"
                
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