"""Main workflow coordinator."""

import logging
import time
import random
from typing import Dict, List, Optional, Set

from omegaconf import DictConfig

from workflow.core.interfaces import WorkflowOrchestrator
from workflow.core.services import RubricService, ScoringService, TieAnalysisService
from workflow.data_structures import WorkflowResult
from workflow.infrastructure import FileHandler, ModelAdapter
from .iteration_manager import IterationManager

log = logging.getLogger(__name__)


class WorkflowCoordinator(WorkflowOrchestrator):
    """Coordinates the entire workflow execution."""
    
    def __init__(self, config: DictConfig):
        """Initialize workflow coordinator.
        
        Args:
            config: Hydra configuration
        """
        self.config = config
        self.file_handler = FileHandler()
        self.iterations = []
        self.final_rubrics_by_prompt = {}
        self.current_rubrics = {}  # Track current rubrics throughout workflow
        self.prompt_resolution_tracker = {}  # Track when each prompt was resolved
        # Track the last iteration at which each rubric was modified. 0 means initial rubrics
        self.rubric_last_modified_iteration = {}
        
        # Initialize services
        self.rubric_service = RubricService()
        self.scoring_service = ScoringService()
        self.tie_analysis_service = TieAnalysisService()
        
        # Initialize model adapters (always use API)
        self.proposer_adapter = ModelAdapter()
        self.verifier_adapter = ModelAdapter()
        
        # Initialize iteration manager with separate adapters
        self.iteration_manager = IterationManager(
            rubric_generator=self.proposer_adapter,
            response_scorer=self.verifier_adapter,
            rubric_service=self.rubric_service,
            scoring_service=self.scoring_service,
            tie_analysis_service=self.tie_analysis_service,
            file_handler=self.file_handler,
            output_dir=config.files.output_dir
        )
    
    def setup(self) -> None:
        """Setup workflow dependencies."""
        # Create output directory
        self.file_handler.create_directory(self.config.files.output_dir)
        
        # Set random seed for reproducibility
        if hasattr(self.config, 'random_seed'):
            random.seed(self.config.random_seed)
    
    def run(self) -> WorkflowResult:
        """Run the complete workflow.
        
        Returns:
            WorkflowResult with final status and iteration information
        """
        start_time = time.time()
        
        try:
            # Resume support: if resume.path is provided, load prior state
            resume_path = getattr(self.config, 'resume', None)
            resume_path = getattr(resume_path, 'path', None) if resume_path else None
            if resume_path:
                log.info(f"Resume mode enabled. Loading state from {resume_path}")
                # Load previously saved config for reference (optional)
                try:
                    prior_cfg_path = f"{resume_path}/config.yaml"
                    if self.file_handler.exists(prior_cfg_path):
                        log.info(f"Found prior run config at {prior_cfg_path}")
                except Exception:
                    pass
                # Load last iteration artifacts if present
                iter_rubrics_files = sorted(self.file_handler.list_files(resume_path, pattern="iteration_*_rubrics.json"))
                iter_scored_files = sorted(self.file_handler.list_files(resume_path, pattern="iteration_*_scored_responses.json"))
                if iter_rubrics_files:
                    last_rubrics_path = str(iter_rubrics_files[-1])
                    log.info(f"Loading latest rubrics from {last_rubrics_path}")
                    try:
                        last_rubrics_blob = self.file_handler.load_json(last_rubrics_path)
                        last_rubrics_list = last_rubrics_blob.get("rubrics", []) if isinstance(last_rubrics_blob, dict) else []
                        self.current_rubrics = {r.get("id"): r for r in last_rubrics_list if isinstance(r, dict) and r.get("id")}
                        for _pid in self.current_rubrics.keys():
                            self.rubric_last_modified_iteration.setdefault(_pid, 0)
                    except Exception as e:
                        log.warning(f"Failed to load last rubrics from {last_rubrics_path}: {e}")
                # Record prior iterations if summary exists
                summary_path = f"{resume_path}/workflow_summary.json"
                if self.file_handler.exists(summary_path):
                    try:
                        summary = self.file_handler.load_json(summary_path)
                        for it in summary.get("iterations", []):
                            try:
                                from workflow.data_structures import IterationInfo
                                info = IterationInfo(
                                    iteration=it.get("iteration", 0),
                                    responses_per_prompt=self.config.workflow.initial_responses_per_prompt,
                                    sample_size=0,
                                    start_time="",
                                )
                                info.status = it.get("status", "resumed")
                                info.has_ties = it.get("has_ties", False)
                                info.prompt_ids_with_ties = []
                                info.prompt_ids_without_ties = []
                                info.ties_per_prompt = {}
                                self.iterations.append(info)
                            except Exception:
                                pass
                    except Exception as e:
                        log.warning(f"Failed to load prior workflow summary: {e}")
                # Keep output_dir as resume_path (engine already set config.files.output_dir)
                log.info("Resume initialization complete")
            # Load initial data
            responses_data = self.file_handler.load_json(
                self.config.files.responses_file)
            log.info(f"Loaded responses data from {self.config.files.responses_file}")
            
            # Convert responses to expected format
            responses = []
            if isinstance(responses_data, list):
                responses = responses_data
            elif isinstance(responses_data, dict):
                # If it's a dict with "responses" key, extract that
                if "responses" in responses_data:
                    responses = responses_data["responses"]
                else:
                    # Otherwise assume it's a dict of responses by ID
                    responses = list(responses_data.values())
            
            log.info(f"Loaded {len(responses)} responses")
            
            # Handle different response formats
            # If responses is a list of prompts with multiple responses each
            if responses and isinstance(responses[0], dict) and 'responses' in responses[0]:
                # Normalize grouped responses: ensure each prompt's responses are strings
                normalized_grouped = []
                for item in responses:
                    try:
                        resp_list = item.get('responses', [])
                        if resp_list and isinstance(resp_list[0], dict):
                            # Convert list of {generator, response} -> list[str]
                            resp_list = [r.get('response', r) for r in resp_list]
                        normalized_grouped.append({
                            'id': item.get('id'),
                            'prompt': item.get('prompt', item.get('id')),
                            'responses': resp_list,
                        })
                    except Exception:
                        # Best-effort normalization; keep original on failure
                        normalized_grouped.append(item)
                responses = normalized_grouped
                # This is the format where each item has id, prompt, and responses list
                all_prompt_ids = [item['id'] for item in responses]
                log.info(f"Found {len(all_prompt_ids)} prompts with multiple responses each")
            else:
                # Handle other formats (individual responses)
                all_prompt_ids = list(set(r.get('prompt_id', r.get('id')) for r in responses))
            
            # Removed mounted workflow support

            # Apply sampling if configured
            if self.config.workflow.sample_size:
                sample_size = self.config.workflow.sample_size
                if len(all_prompt_ids) > sample_size:
                    prompts_to_process = set(random.sample(all_prompt_ids, sample_size))
                    log.info(f"Sampled {sample_size} prompts from {len(all_prompt_ids)} available prompts")
                else:
                    prompts_to_process = set(all_prompt_ids)
                    log.info(f"Using all {len(all_prompt_ids)} prompts (less than sample size {sample_size})")
                
                # Filter responses to only include sampled prompts
                if responses and isinstance(responses[0], dict) and 'responses' in responses[0]:
                    current_responses = [r for r in responses if r['id'] in prompts_to_process]
                else:
                    current_responses = [r for r in responses if r.get('prompt_id', r.get('id')) in prompts_to_process]
            else:
                prompts_to_process = set(all_prompt_ids)
                current_responses = responses
                log.info(f"Processing all {len(all_prompt_ids)} prompts")
            
            # Removed mounted workflow support

            # Initialize or load rubrics
            if self.config.files.initial_rubrics:
                rubrics_data = self.file_handler.load_json(
                    self.config.files.initial_rubrics)
                # Convert to dict format
                if isinstance(rubrics_data, dict) and "rubrics" in rubrics_data:
                    current_rubrics = {}
                    for rubric in rubrics_data["rubrics"]:
                        current_rubrics[rubric["id"]] = rubric
                else:
                    current_rubrics = rubrics_data
                log.info(f"Loaded {len(current_rubrics)} initial rubrics")
                # If some prompts lack rubrics or have empty criteria, generate them now
                try:
                    # Build a mapping of prompt_id -> prompt text for the prompts we'll process
                    prompt_text_by_id = {}
                    if current_responses and isinstance(current_responses, list):
                        if isinstance(current_responses[0], dict) and 'responses' in current_responses[0]:
                            # Grouped format: each item has id, prompt, responses
                            for item in current_responses:
                                pid = item.get('id')
                                if pid is not None:
                                    prompt_text_by_id[pid] = item.get('prompt', str(pid))
                        else:
                            # Individual responses format: collect first seen prompt text per id
                            for resp in current_responses:
                                pid = resp.get('prompt_id', resp.get('id'))
                                if pid is None:
                                    continue
                                if pid not in prompt_text_by_id:
                                    prompt_text_by_id[pid] = resp.get('prompt', str(pid))
                    # Determine which prompt ids we are processing in this run
                    prompt_ids_in_scope: Set[str] = set(prompt_text_by_id.keys())
                    # Find missing or empty rubrics among prompts in scope
                    missing_prompt_ids: List[str] = []
                    for pid in prompt_ids_in_scope:
                        r = current_rubrics.get(pid)
                        if not r or not isinstance(r, dict) or not r.get('criteria'):
                            missing_prompt_ids.append(pid)
                    if missing_prompt_ids:
                        log.info(f"Found {len(missing_prompt_ids)} prompts with missing/empty rubrics. Generating now...")
                        # Prepare minimal prompt objects for generation
                        prompts_for_generation = [
                            {"id": pid, "prompt": prompt_text_by_id.get(pid, str(pid))}
                            for pid in missing_prompt_ids
                        ]
                        # Generate rubrics for missing prompts using the configured generator
                        generated_missing = self.proposer_adapter.generate_initial_rubrics(
                            prompts_for_generation,
                            self.config.models.rubric_generator
                        )
                        # Merge into current rubrics
                        if isinstance(generated_missing, dict):
                            current_rubrics.update(generated_missing)
                            log.info(f"Generated and merged {len(generated_missing)} missing rubrics")
                        # If some prompts are still missing (e.g., model failed to parse), backfill with defaults
                        still_missing = [pid for pid in missing_prompt_ids if pid not in current_rubrics or not current_rubrics.get(pid, {}).get('criteria')]
                        if still_missing:
                            log.warning(f"Backfilling {len(still_missing)} prompts with default fallback rubrics")
                            # Default criteria spec aligned with utils.extract_rubric_from_loaded_data fallback
                            default_criteria = [
                                {"local_id": "c1",  "criterion": "Instruction Following — addresses all explicit and implicit requirements in the prompt.", "weight": 3},
                                {"local_id": "c2",  "criterion": "Constraint Compliance — follows any format, length, or content constraints specified.", "weight": 3},
                                {"local_id": "c3",  "criterion": "Truthfulness — central and supporting claims are factually correct and non-misleading.", "weight": 3},
                                {"local_id": "c4",  "criterion": "Use of Sources — cites or attributes sources appropriately when needed; avoids fabricated citations.", "weight": 1},
                                {"local_id": "c5",  "criterion": "Completeness — covers all key aspects of the task with sufficient detail.", "weight": 3},
                                {"local_id": "c6",  "criterion": "Reasoning Quality — demonstrates clear, logical reasoning and, when applicable, shows steps.", "weight": 1},
                                {"local_id": "c7",  "criterion": "Safety & Policy — avoids harmful, disallowed, or unsafe content; includes caveats where appropriate.", "weight": 1},
                                {"local_id": "c8",  "criterion": "Presentation — organized, readable, and well-formatted (headings, bullets, code blocks as needed).", "weight": 2},
                                {"local_id": "c9",  "criterion": "Clarity & Concision — clear, unambiguous writing without unnecessary verbosity.", "weight": 2},
                                {"local_id": "c10", "criterion": "Tone & Helpfulness — professional, helpful, and appropriate tone for the user’s request.", "weight": 2}
                            ]
                            for pid in still_missing:
                                prompt_text = prompt_text_by_id.get(pid, str(pid))
                                current_rubrics[pid] = {
                                    "id": pid,
                                    "prompt": prompt_text,
                                    "original_rubric": "",  # no model-generated rubric text
                                    "total_criteria": len(default_criteria),
                                    "criteria": default_criteria,
                                    "total_weight": sum(c.get("weight", 0) for c in default_criteria)
                                }
                            log.info(f"Backfilled {len(still_missing)} default rubrics")
                        # Save the augmented initial rubrics into the run directory
                        self._save_initial_rubrics(current_rubrics)
                except Exception as gen_missing_err:
                    log.warning(f"Failed to generate missing rubrics: {gen_missing_err}")
                # Store initial rubrics
                self.current_rubrics = current_rubrics
            else:
                log.info("Generating initial rubrics from responses")
                sampled = self._sample_responses_for_rubrics(current_responses)
                current_rubrics = self.proposer_adapter.generate_initial_rubrics(
                    sampled, self.config.models.rubric_generator)
                self._save_initial_rubrics(current_rubrics)
            
            # Store initial rubrics
            self.current_rubrics = current_rubrics
            # Initialize last-modified iteration mapping to 0 for all prompts present in initial rubrics
            for _pid in self.current_rubrics.keys():
                self.rubric_last_modified_iteration.setdefault(_pid, 0)
            

            # Main iteration loop
            for iteration_num in range(1, self.config.workflow.max_iterations + 1):
                log.info(f"\n{'='*60}")
                log.info(f"Starting iteration {iteration_num}")
                log.info(f"{'='*60}")
                
                # Convert responses to dictionary format expected by iteration manager
                responses_dict = {}
                if isinstance(current_responses, dict):
                    # Already in dictionary format (from previous iteration)
                    responses_dict = current_responses
                elif current_responses and isinstance(current_responses[0], dict) and 'responses' in current_responses[0]:
                    # Current responses are in grouped format. Normalize to ensure responses are strings.
                    for item in current_responses:
                        resp_list = item.get('responses', [])
                        if resp_list and isinstance(resp_list[0], dict):
                            resp_list = [r.get('response', r) for r in resp_list]
                        responses_dict[item['id']] = {
                            'id': item['id'],
                            'prompt': item.get('prompt', item['id']),
                            'responses': resp_list,
                        }
                else:
                    # Group individual responses by prompt_id
                    for response in current_responses:
                        prompt_id = response.get('prompt_id', response.get('id'))
                        if prompt_id not in responses_dict:
                            responses_dict[prompt_id] = {
                                'id': prompt_id,
                                'prompt': response.get('prompt', prompt_id),
                                'responses': []
                            }
                        responses_dict[prompt_id]['responses'].append(response.get('response', response))
                
                log.info(f"Prepared {len(responses_dict)} prompts for iteration")
                
                # Only process prompts that still have responses
                current_prompt_ids = list(responses_dict.keys())

                log.info(f"Processing {len(current_prompt_ids)} prompts with responses")
                
                # Check if we have any prompts left to process
                if not current_prompt_ids:
                    log.warning("No prompts with responses left to process. All prompts have been resolved.")
                    break
                

                # Run iteration
                iter_info, tie_analysis, updated_rubrics, next_responses = \
                    self.iteration_manager.run_iteration(
                        iteration_num=iteration_num,
                        responses=responses_dict,
                        current_rubrics=current_rubrics,
                        prompt_ids=current_prompt_ids,
                        config={
                            'verifier_model': self.config.models.verifier,
                            'rubric_model': self.config.models.rubric_generator,
                            'responses_per_prompt': self.config.workflow.initial_responses_per_prompt,
                            'skip_scoring': self.config.grading.skip_scoring
                        }
                    )
                
                # Store iteration info
                self.iterations.append(iter_info)

                # Track which rubrics changed in this iteration and record last modified iteration
                try:
                    for _pid, _new_rubric in updated_rubrics.items():
                        _old_rubric = current_rubrics.get(_pid)
                        if _old_rubric != _new_rubric:
                            self.rubric_last_modified_iteration[_pid] = iteration_num
                except Exception:
                    # Best-effort tracking; do not fail workflow due to comparison issues
                    pass
                
                # With force_continue always true, log but don't mark as resolved
                for prompt_id in tie_analysis.prompt_ids_without_ties:
                    highest_score = tie_analysis.highest_scores_by_prompt.get(prompt_id, None)
                    log.info(f"🔄 Prompt {prompt_id} has no ties at highest score {highest_score}, but continuing due to force_continue")
                
                # Check termination conditions (force_continue is always True)
                should_continue, reason = self.tie_analysis_service.should_continue_iteration(
                    tie_analysis, 
                    self.config.grading.skip_scoring
                )
                
                if not should_continue:
                    iter_info.status = reason
                    log.info(f"Workflow terminating: {reason}")
                    break
                

                # Update for next iteration
                current_rubrics = updated_rubrics
                current_responses = next_responses
            
            # Store current rubrics for final save
            self.current_rubrics = current_rubrics
            
            # Save final outputs
            self._save_final_outputs()
            
        finally:
            # Cleanup resources - this always runs even if there's an error
            log.info("Cleaning up workflow resources")
            if hasattr(self, 'proposer_adapter') and hasattr(self.proposer_adapter, 'cleanup'):
                self.proposer_adapter.cleanup()
            if hasattr(self, 'verifier_adapter') and hasattr(self.verifier_adapter, 'cleanup'):
                self.verifier_adapter.cleanup()
        
        # Determine final status
        final_status = self.iterations[-1].status if self.iterations else "no_iterations"
        
        return WorkflowResult(
            status=final_status,
            total_time=time.time() - start_time,
            iterations=self.iterations,
            final_rubrics_by_prompt=self.final_rubrics_by_prompt,
            prompt_resolution_tracker=self.prompt_resolution_tracker
        )
    
    def _load_initial_data(self) -> tuple[Dict, List[str]]:
        """Load initial response data and determine prompts to process.
        
        Returns:
            Tuple of (responses_data, prompts_to_process)
        """
        # Load responses
        responses_data = self.file_handler.load_json(self.config.files.responses_file)
        
        # Convert to internal format if needed
        if isinstance(responses_data, list):
            # Convert list format to dict format
            responses_dict = {}
            for response_obj in responses_data:
                responses_dict[response_obj["id"]] = response_obj
            responses_data = responses_dict
        elif isinstance(responses_data, dict) and "responses" in responses_data:
            # Extract responses from wrapper
            responses_data = responses_data["responses"]
        
        # No mounted workflow support
        processed_prompts = set()
        
        # Determine prompts to process
        available_prompts = [
            p for p in responses_data.keys()
            if p not in processed_prompts
        ]
        
        # Sample if needed
        if self.config.workflow.sample_size:
            sample_size = self.config.workflow.sample_size
            if len(available_prompts) > sample_size:
                prompts_to_process = random.sample(available_prompts, sample_size)
            else:
                prompts_to_process = available_prompts
        else:
            prompts_to_process = available_prompts
        
        return responses_data, prompts_to_process
    
    # Removed mounted workflow support
    # def _load_mounted_workflow(self) -> Set[str]:
    #     """Load processed prompts from mounted workflow.
    #     
    #     Returns:
    #         Set of processed prompt IDs
    #     """
    #     log.info(f"Loading mounted workflow from {self.config.mounted_workflow.path}")
    #     
    #     # Try to load workflow summary
    #     summary_file = f"{self.config.mounted_workflow.path}/workflow_summary.json"
        if self.file_handler.exists(summary_file):
            summary_data = self.file_handler.load_json(summary_file)
            # Extract processed prompts from summary
            # This is a simplified version - would need more logic for real implementation
            return set()
        
        return set()
    
    def _get_initial_rubrics(
        self,
        responses_data: Dict,
        prompts_to_process: List[str]
    ) -> Dict[str, Dict]:
        """Get or generate initial rubrics.
        
        Args:
            responses_data: All response data
            prompts_to_process: List of prompt IDs to process
            
        Returns:
            Dictionary of rubrics by prompt_id
        """
        # Check if initial rubrics file is provided
        if self.config.files.initial_rubrics:
            log.info(f"Loading initial rubrics from {self.config.files.initial_rubrics}")
            rubrics_data = self.file_handler.load_json(self.config.files.initial_rubrics)
            
            # Convert to dict format
            if isinstance(rubrics_data, dict) and "rubrics" in rubrics_data:
                rubrics_dict = {}
                for rubric in rubrics_data["rubrics"]:
                    rubrics_dict[rubric["id"]] = rubric
                return rubrics_dict
            else:
                return rubrics_data
        
        # Generate initial rubrics
        log.info("Generating initial rubrics...")
        
        # Sample 2 responses per prompt
        sampled_responses = []
        for prompt_id in prompts_to_process:
            response_obj = responses_data[prompt_id]
            
            if "responses" in response_obj and len(response_obj["responses"]) >= 2:
                sampled = random.sample(response_obj["responses"], 2)
                sampled_responses.append({
                    "id": prompt_id,
                    "prompt": response_obj.get("prompt", prompt_id),
                    "responses": sampled
                })
        
        if not sampled_responses:
            raise RuntimeError("No prompts have enough responses for rubric generation")
        
        # Generate rubrics
        return self.model_adapter.generate_initial_rubrics(
            responses=sampled_responses,
            model=self.config.models.rubric_generator
        )
    
    def _save_final_outputs(self) -> None:
        """Save final workflow outputs."""
        output_dir = self.config.files.output_dir
        
        # Combine final rubrics - include resolved prompts and add latest rubrics for unresolved ones
        all_final_rubrics = self.final_rubrics_by_prompt.copy()
        
        # Add any prompts that still have ties using their latest rubrics
        for prompt_id, rubric in self.current_rubrics.items():
            if prompt_id not in all_final_rubrics:
                all_final_rubrics[prompt_id] = rubric
        
        # Save final rubrics
        if all_final_rubrics:
            # Inject per-rubric last_modified_iteration information
            _final_rubrics_list = []
            for _pid, _rubric in all_final_rubrics.items():
                _rubric_with_meta = _rubric.copy()
                # Remove large/unnecessary fields before final save
                if "original_rubric" in _rubric_with_meta:
                    _rubric_with_meta.pop("original_rubric", None)
                if "response_selection_info" in _rubric_with_meta:
                    _rubric_with_meta.pop("response_selection_info", None)
                _rubric_with_meta["last_modified_iteration"] = self.rubric_last_modified_iteration.get(_pid, 0)
                _final_rubrics_list.append(_rubric_with_meta)

            final_rubrics = {
                "metadata": {
                    "total_rubrics": len(all_final_rubrics),
                    "prompts_resolved": len(self.final_rubrics_by_prompt),
                    "prompts_with_ties": len(all_final_rubrics) - len(self.final_rubrics_by_prompt),
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
                },
                "rubrics": _final_rubrics_list
            }
            
            self.file_handler.save_json(
                final_rubrics,
                f"{output_dir}/final_rubrics.json"
            )
        
        # Save workflow summary
        summary = {
            "total_iterations": len(self.iterations),
            "total_prompts_processed": len(all_final_rubrics),
            "prompts_resolved": len(self.final_rubrics_by_prompt),
            "prompts_with_ties": len(all_final_rubrics) - len(self.final_rubrics_by_prompt),
            "prompt_resolution_details": self.prompt_resolution_tracker,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "status": it.status,
                    "has_ties": it.has_ties,
                    "tied_prompts": len(it.prompt_ids_with_ties),
                    "resolved_prompts": len(it.prompt_ids_without_ties),
                    "newly_resolved_prompts": [
                        pid for pid in it.prompt_ids_without_ties 
                        if self.prompt_resolution_tracker.get(pid, {}).get('resolved_at_iteration') == it.iteration
                    ]
                }
                for it in self.iterations
            ]
        }
        
        self.file_handler.save_json(
            summary,
            f"{output_dir}/workflow_summary.json"
        )
        
        # Save detailed prompt resolution tracking
        resolution_report = {
            "metadata": {
                "total_prompts": len(all_final_rubrics),
                "resolved_prompts": len(self.prompt_resolution_tracker),
                "unresolved_prompts": len(all_final_rubrics) - len(self.prompt_resolution_tracker),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "resolutions": self.prompt_resolution_tracker,
            "unresolved_prompt_ids": [
                pid for pid in all_final_rubrics.keys() 
                if pid not in self.prompt_resolution_tracker
            ]
        }
        
        self.file_handler.save_json(
            resolution_report,
            f"{output_dir}/prompt_resolution_report.json"
        )
        
        log.info(f"Saved final outputs to {output_dir}")
    
    def _create_result(self, status: str, start_time: float) -> WorkflowResult:
        """Create workflow result object.
        
        Args:
            status: Final workflow status
            start_time: Workflow start time
            
        Returns:
            WorkflowResult object
        """
        total_time = time.time() - start_time
        
        return WorkflowResult(
            status=status,
            total_time=total_time,
            iterations=self.iterations,
            final_rubrics_by_prompt=self.final_rubrics_by_prompt,
            processed_prompt_ids=set(self.final_rubrics_by_prompt.keys()),
            prompt_resolution_tracker=self.prompt_resolution_tracker
        ) 
    
    def _sample_responses_for_rubrics(self, responses: List[Dict]) -> List[Dict]:
        """Sample responses for rubric generation.
        
        Args:
            responses: List of all responses
            
        Returns:
            List of sampled responses (2 per prompt)
        """
        log.debug(f"_sample_responses_for_rubrics called with {len(responses)} items")
        
        # Check if responses are already in the format with multiple responses per prompt
        if responses and isinstance(responses[0], dict) and 'responses' in responses[0]:
            # Already in the right format, just sample 2 responses from each
            sampled = []
            for item in responses:
                # Normalize any object responses to strings
                resp_list = item.get('responses', [])
                if resp_list and isinstance(resp_list[0], dict):
                    resp_list = [r.get('response', r) for r in resp_list]
                if len(resp_list) >= 2:
                    sampled_responses = random.sample(resp_list, 2)
                    sampled.append({
                        'id': item['id'],
                        'prompt': item['prompt'],
                        'responses': sampled_responses
                    })
            log.info(f"Sampled 2 responses each for {len(sampled)} prompts for rubric generation")
            return sampled
        
        # Otherwise, group responses by prompt
        responses_by_prompt = {}
        for response in responses:
            prompt_id = response.get('prompt_id', response.get('id'))
            if prompt_id not in responses_by_prompt:
                responses_by_prompt[prompt_id] = {
                    'id': prompt_id,
                    'prompt': response.get('prompt', prompt_id),
                    'responses': []
                }
            # Add the response text
            if 'response' in response:
                responses_by_prompt[prompt_id]['responses'].append(response['response'])
            elif 'responses' in response and isinstance(response['responses'], list):
                responses_by_prompt[prompt_id]['responses'].extend(response['responses'])
        
        log.debug(f"Grouped into {len(responses_by_prompt)} prompts")
        
        # Sample 2 responses per prompt
        sampled = []
        for prompt_id, prompt_data in responses_by_prompt.items():
            if len(prompt_data['responses']) >= 2:
                sampled_responses = random.sample(prompt_data['responses'], 2)
                sampled.append({
                    'id': prompt_id,
                    'prompt': prompt_data['prompt'],
                    'responses': sampled_responses
                })
        
        log.info(f"Sampled 2 responses each for {len(sampled)} prompts for rubric generation")
        return sampled
    
    def _save_initial_rubrics(self, rubrics: Dict[str, Dict]) -> None:
        """Save initial rubrics to file.
        
        Args:
            rubrics: Dictionary of rubrics by prompt_id
        """
        output_dir = self.config.files.output_dir
        rubrics_data = {
            "metadata": {
                "total_rubrics": len(rubrics),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "source": "generated"
            },
            "rubrics": list(rubrics.values())
        }
        
        self.file_handler.save_json(
            rubrics_data,
            f"{output_dir}/initial_rubrics.json"
        )
        log.info(f"Saved {len(rubrics)} initial rubrics to {output_dir}/initial_rubrics.json") 