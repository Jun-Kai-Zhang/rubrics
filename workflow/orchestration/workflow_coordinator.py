"""Main workflow coordinator."""

import logging
import time
import random
from typing import Dict, List, Set

from omegaconf import DictConfig, OmegaConf

from workflow.core.interfaces import WorkflowOrchestrator
from workflow.core.services import RubricService, TieAnalysisService
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
        self.tie_analysis_service = TieAnalysisService()
        
        # Initialize separate model adapters for proposer (rubrics) and verifier (scoring)
        proposer_backend = config.models.proposer_backend
        verifier_backend = config.models.verifier_backend
        api_config = OmegaConf.to_container(config.api, resolve=True) if hasattr(config, 'api') else {}

        proposer_gpu_config = {
            'tensor_parallel_size': config.vllm.tensor_parallel_size,
            'max_model_len': config.vllm.max_model_len,
            'enforce_eager': config.vllm.enforce_eager,
        } if proposer_backend == "vllm" else None

        verifier_gpu_config = {
            'tensor_parallel_size': config.vllm.tensor_parallel_size,
            'max_model_len': config.vllm.max_model_len,
            'enforce_eager': config.vllm.enforce_eager,
        } if verifier_backend == "vllm" else None

        self.proposer_adapter = ModelAdapter(
            backend=proposer_backend,
            gpu_config=proposer_gpu_config,
            api_config=api_config
        )

        self.verifier_adapter = ModelAdapter(
            backend=verifier_backend,
            gpu_config=verifier_gpu_config,
            api_config=api_config
        )
        
        # Initialize iteration manager with separate adapters
        self.iteration_manager = IterationManager(
            rubric_generator=self.proposer_adapter,
            response_scorer=self.verifier_adapter,
            rubric_service=self.rubric_service,
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
        current_responses: Dict[str, Dict] = {}
        termination_reason = "no_prompts_available"
        
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
            
            # Load and normalize responses
            responses_data = self.file_handler.load_json(self.config.files.responses_file)
            log.info(f"Loaded responses data from {self.config.files.responses_file}")
            normalized_responses = self._normalize_responses(responses_data)
            log.info(f"Prepared responses for {len(normalized_responses)} prompts")

            # Apply sampling if configured
            prompts_to_process = list(normalized_responses.keys())
            if self.config.workflow.sample_size:
                sample_size = self.config.workflow.sample_size
                if len(prompts_to_process) > sample_size:
                    prompts_to_process = random.sample(prompts_to_process, sample_size)
                    log.info(f"Sampled {sample_size} prompts from {len(normalized_responses)} available prompts")
                else:
                    log.info(f"Using all {len(prompts_to_process)} prompts (less than configured sample size)")
                normalized_responses = {
                    pid: normalized_responses[pid]
                    for pid in prompts_to_process
                }
            else:
                log.info(f"Processing all {len(prompts_to_process)} prompts")

            current_responses = self._limit_initial_responses(normalized_responses)
            if not current_responses:
                log.warning("No prompts have at least two responses after preprocessing.")

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
                    prompt_text_by_id = {
                        pid: data.get('prompt', str(pid))
                        for pid, data in current_responses.items()
                    }
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
                log.info("Generating initial rubrics from prompts")
                sampled = self._prepare_prompts_for_initial_rubrics(current_responses)
                current_rubrics = self.proposer_adapter.generate_initial_rubrics(
                    sampled, self.config.models.rubric_generator)
                self._save_initial_rubrics(current_rubrics)
            
            # Store initial rubrics
            self.current_rubrics = current_rubrics
            # Initialize last-modified iteration mapping to 0 for all prompts present in initial rubrics
            for _pid in self.current_rubrics.keys():
                self.rubric_last_modified_iteration.setdefault(_pid, 0)

            # Main iteration loop
            if not current_responses:
                termination_reason = "insufficient_responses"
            else:
                termination_reason = "max_iterations_reached"
                for iteration_num in range(1, self.config.workflow.max_iterations + 1):
                    log.info(f"\n{'='*60}")
                    log.info(f"Starting iteration {iteration_num}")
                    log.info(f"{'='*60}")
                    
                    responses_dict = current_responses
                    current_prompt_ids = list(responses_dict.keys())
                    if not current_prompt_ids:
                        log.warning("No prompts with responses left to process.")
                        termination_reason = "responses_exhausted"
                        break
                    
                    log.info(f"Processing {len(current_prompt_ids)} prompts with responses")
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
                                'selection_strategy': self.config.workflow.get('selection_strategy', 'top2'),
                                'exclude_used_reference_responses': self.config.workflow.get('exclude_used_reference_responses', False)
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
                        pass
                    
                    # Determine which prompts dropped out of the active set
                    resolved_prompt_ids = set(responses_dict.keys()) - set(next_responses.keys())
                    for prompt_id in resolved_prompt_ids:
                        if prompt_id not in self.prompt_resolution_tracker:
                            highest_score = tie_analysis.highest_scores_by_prompt.get(prompt_id, None)
                            self.prompt_resolution_tracker[prompt_id] = {
                                'resolved_at_iteration': iteration_num,
                                'highest_score': highest_score,
                                'reason': 'responses_exhausted'
                            }
                            log.info(f"✅ Prompt {prompt_id} resolved at iteration {iteration_num}")

                    current_rubrics = updated_rubrics
                    current_responses = next_responses
                    
                    if not current_responses:
                        termination_reason = "responses_exhausted"
                        break
            
            # Store current rubrics for final save
            self.current_rubrics = current_rubrics
            self.final_rubrics_by_prompt = current_rubrics.copy()
            
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
        final_status = termination_reason
        
        return WorkflowResult(
            status=final_status,
            total_time=time.time() - start_time,
            iterations=self.iterations,
            final_rubrics_by_prompt=self.final_rubrics_by_prompt,
            prompt_resolution_tracker=self.prompt_resolution_tracker
        )
    
    
    def _save_final_outputs(self) -> None:
        """Save final workflow outputs."""
        output_dir = self.config.files.output_dir
        
        # Combine final rubrics from the latest iteration
        all_final_rubrics = self.current_rubrics.copy()
        
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
                    "prompts_resolved": len(self.prompt_resolution_tracker),
                    "remaining_prompts": len(all_final_rubrics) - len(self.prompt_resolution_tracker),
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
            "prompts_resolved": len(self.prompt_resolution_tracker),
            "prompts_remaining": len(all_final_rubrics) - len(self.prompt_resolution_tracker),
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
    
    def _prepare_prompts_for_initial_rubrics(self, responses: Dict[str, Dict]) -> List[Dict]:
        """Prepare prompt payload used for initial rubric generation."""
        payload = []
        for prompt_id, data in responses.items():
            payload.append({
                "id": prompt_id,
                "prompt": data.get("prompt", str(prompt_id))
            })
        return payload
    
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

    def _normalize_responses(self, responses_data) -> Dict[str, Dict]:
        """Normalize various response payloads into a prompt-centric dictionary."""
        normalized: Dict[str, Dict] = {}

        if isinstance(responses_data, dict):
            if "responses" in responses_data and isinstance(responses_data["responses"], list):
                responses_data = responses_data["responses"]
            else:
                responses_data = list(responses_data.values())

        if isinstance(responses_data, list) and responses_data:
            if isinstance(responses_data[0], dict) and 'responses' in responses_data[0]:
                for item in responses_data:
                    prompt_id = item.get('id')
                    if prompt_id is None:
                        continue
                    resp_list = item.get('responses', [])
                    normalized[prompt_id] = {
                        'id': prompt_id,
                        'prompt': item.get('prompt', str(prompt_id)),
                        'responses': self._coerce_responses(resp_list)
                    }
            else:
                for response in responses_data:
                    prompt_id = response.get('prompt_id', response.get('id'))
                    if prompt_id is None:
                        continue
                    normalized.setdefault(prompt_id, {
                        'id': prompt_id,
                        'prompt': response.get('prompt', str(prompt_id)),
                        'responses': []
                    })
                    normalized[prompt_id]['responses'].append(response.get('response', response))
        return normalized

    def _coerce_responses(self, response_list: List) -> List[str]:
        """Ensure response values are simple strings."""
        coerced = []
        for resp in response_list:
            if isinstance(resp, dict):
                coerced.append(resp.get('response', str(resp)))
            else:
                coerced.append(resp)
        return coerced

    def _limit_initial_responses(self, responses: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply the initial_responses_per_prompt cap and prune under-populated prompts."""
        max_responses = getattr(self.config.workflow, 'initial_responses_per_prompt', None)
        filtered: Dict[str, Dict] = {}
        skipped = 0
        for prompt_id, data in responses.items():
            resp_list = data.get('responses', [])
            if max_responses and max_responses > 0:
                resp_list = resp_list[:max_responses]
            if len(resp_list) < 2:
                skipped += 1
                continue
            filtered[prompt_id] = {
                'id': prompt_id,
                'prompt': data.get('prompt', str(prompt_id)),
                'responses': resp_list
            }
        if skipped:
            log.warning(f"Skipped {skipped} prompts with fewer than 2 responses.")
        return filtered
