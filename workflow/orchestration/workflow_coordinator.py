"""Main workflow coordinator."""

import json
import logging
import time
import random
from typing import Dict, List, Set

from omegaconf import DictConfig, OmegaConf

from workflow.core.interfaces import WorkflowOrchestrator
from workflow.core.services import RubricService
from workflow.data_structures import WorkflowResult
from workflow.infrastructure import FileHandler, ModelAdapter
from .iteration_manager import IterationManager

log = logging.getLogger(__name__)


class WorkflowCoordinator(WorkflowOrchestrator):
    """Coordinates the entire workflow execution."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.file_handler = FileHandler()
        self.iterations = []
        self.final_rubrics_by_prompt = {}
        self.current_rubrics = {}
        self.prompt_resolution_tracker = {}
        self.rubric_last_modified_iteration = {}

        # Initialize services
        self.rubric_service = RubricService()

        # Build adapter configs from Hydra config
        api_config = (
            OmegaConf.to_container(config.api, resolve=True)
            if hasattr(config, "api")
            else {}
        )
        vllm_config = (
            OmegaConf.to_container(config.vllm, resolve=True)
            if hasattr(config, "vllm")
            else {}
        )

        proposer_backend = config.models.proposer_backend
        verifier_backend = config.models.verifier_backend

        # When both proposer and verifier use vLLM, assign different ports
        proposer_vllm = dict(vllm_config) if proposer_backend == "vllm" else None
        verifier_vllm = dict(vllm_config) if verifier_backend == "vllm" else None
        if proposer_vllm and verifier_vllm:
            verifier_vllm["port"] = verifier_vllm.get("port", 8000) + 1

        self.proposer_adapter = ModelAdapter(
            backend=proposer_backend,
            vllm_config=proposer_vllm,
            api_config=api_config,
        )
        self.verifier_adapter = ModelAdapter(
            backend=verifier_backend,
            vllm_config=verifier_vllm,
            api_config=api_config,
        )

        self.iteration_manager = IterationManager(
            rubric_generator=self.proposer_adapter,
            response_scorer=self.verifier_adapter,
            rubric_service=self.rubric_service,
            file_handler=self.file_handler,
            output_dir=config.files.output_dir,
        )

    def setup(self) -> None:
        """Setup workflow dependencies."""
        self.file_handler.create_directory(self.config.files.output_dir)
        if hasattr(self.config, "random_seed"):
            random.seed(self.config.random_seed)

    def run(self) -> WorkflowResult:
        """Run the complete workflow."""
        start_time = time.time()
        current_responses: Dict[str, Dict] = {}
        termination_reason = "no_prompts_available"

        try:
            # Resume support
            resume_path = getattr(self.config, "resume", None)
            resume_path = getattr(resume_path, "path", None) if resume_path else None
            if resume_path:
                log.info(f"Resume mode enabled. Loading state from {resume_path}")
                try:
                    prior_cfg_path = f"{resume_path}/config.yaml"
                    if self.file_handler.exists(prior_cfg_path):
                        log.info(f"Found prior run config at {prior_cfg_path}")
                except Exception:
                    pass

                iter_rubrics_files = sorted(
                    self.file_handler.list_files(
                        resume_path, pattern="iteration_*_rubrics.json"
                    )
                )
                if iter_rubrics_files:
                    last_rubrics_path = str(iter_rubrics_files[-1])
                    log.info(f"Loading latest rubrics from {last_rubrics_path}")
                    try:
                        last_rubrics_blob = self.file_handler.load_json(
                            last_rubrics_path
                        )
                        last_rubrics_list = (
                            last_rubrics_blob
                            if isinstance(last_rubrics_blob, list)
                            else []
                        )
                        self.current_rubrics = {
                            r.get("id"): r
                            for r in last_rubrics_list
                            if isinstance(r, dict) and r.get("id")
                        }
                        for _pid in self.current_rubrics:
                            self.rubric_last_modified_iteration.setdefault(_pid, 0)
                    except Exception as e:
                        log.warning(
                            f"Failed to load last rubrics from {last_rubrics_path}: {e}"
                        )

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
                                self.iterations.append(info)
                            except Exception:
                                pass
                    except Exception as e:
                        log.warning(
                            f"Failed to load prior workflow summary: {e}"
                        )
                log.info("Resume initialization complete")

            # Load and normalize responses
            responses_data = self.file_handler.load_json(
                self.config.files.responses_file
            )
            log.info(
                f"Loaded responses data from {self.config.files.responses_file}"
            )
            normalized_responses = self._normalize_responses(responses_data)
            log.info(f"Prepared responses for {len(normalized_responses)} prompts")

            # Apply sampling if configured
            prompts_to_process = list(normalized_responses.keys())
            if self.config.workflow.sample_size:
                sample_size = self.config.workflow.sample_size
                if len(prompts_to_process) > sample_size:
                    prompts_to_process = random.sample(
                        prompts_to_process, sample_size
                    )
                    log.info(
                        f"Sampled {sample_size} prompts from "
                        f"{len(normalized_responses)} available prompts"
                    )
                else:
                    log.info(
                        f"Using all {len(prompts_to_process)} prompts "
                        "(less than configured sample size)"
                    )
                normalized_responses = {
                    pid: normalized_responses[pid] for pid in prompts_to_process
                }
            else:
                log.info(f"Processing all {len(prompts_to_process)} prompts")

            current_responses = self._limit_initial_responses(normalized_responses)
            if not current_responses:
                log.warning(
                    "No prompts have at least two responses after preprocessing."
                )

            # Initialize or load rubrics
            if self.config.files.initial_rubrics:
                rubrics_data = self.file_handler.load_json(
                    self.config.files.initial_rubrics
                )
                if isinstance(rubrics_data, list):
                    current_rubrics = {
                        r["id"]: r for r in rubrics_data
                    }
                else:
                    current_rubrics = rubrics_data
                log.info(f"Loaded {len(current_rubrics)} initial rubrics")

                # Fill in missing rubrics
                try:
                    prompt_text_by_id = {
                        pid: data.get("prompt", str(pid))
                        for pid, data in current_responses.items()
                    }
                    prompt_ids_in_scope: Set[str] = set(prompt_text_by_id.keys())
                    missing_prompt_ids: List[str] = [
                        pid
                        for pid in prompt_ids_in_scope
                        if not current_rubrics.get(pid)
                        or not isinstance(current_rubrics.get(pid), dict)
                        or not current_rubrics[pid].get("criteria")
                    ]
                    if missing_prompt_ids:
                        log.info(
                            f"Found {len(missing_prompt_ids)} prompts with "
                            "missing/empty rubrics. Generating now..."
                        )
                        prompts_for_generation = [
                            {
                                "id": pid,
                                "prompt": prompt_text_by_id.get(pid, str(pid)),
                            }
                            for pid in missing_prompt_ids
                        ]
                        generated_missing = (
                            self.proposer_adapter.generate_initial_rubrics(
                                prompts_for_generation,
                                self.config.models.rubric_generator,
                            )
                        )
                        if isinstance(generated_missing, dict):
                            current_rubrics.update(generated_missing)
                            log.info(
                                f"Generated and merged "
                                f"{len(generated_missing)} missing rubrics"
                            )
                        still_missing = [
                            pid
                            for pid in missing_prompt_ids
                            if pid not in current_rubrics
                            or not current_rubrics.get(pid, {}).get("criteria")
                        ]
                        if still_missing:
                            log.warning(
                                f"Backfilling {len(still_missing)} prompts "
                                "with default fallback rubrics"
                            )
                            default_criteria = [
                                {"local_id": "c1", "criterion": "Instruction Following — addresses all explicit and implicit requirements in the prompt.", "weight": 3},
                                {"local_id": "c2", "criterion": "Constraint Compliance — follows any format, length, or content constraints specified.", "weight": 3},
                                {"local_id": "c3", "criterion": "Truthfulness — central and supporting claims are factually correct and non-misleading.", "weight": 3},
                                {"local_id": "c4", "criterion": "Use of Sources — cites or attributes sources appropriately when needed; avoids fabricated citations.", "weight": 1},
                                {"local_id": "c5", "criterion": "Completeness — covers all key aspects of the task with sufficient detail.", "weight": 3},
                                {"local_id": "c6", "criterion": "Reasoning Quality — demonstrates clear, logical reasoning and, when applicable, shows steps.", "weight": 1},
                                {"local_id": "c7", "criterion": "Safety & Policy — avoids harmful, disallowed, or unsafe content; includes caveats where appropriate.", "weight": 1},
                                {"local_id": "c8", "criterion": "Presentation — organized, readable, and well-formatted (headings, bullets, code blocks as needed).", "weight": 2},
                                {"local_id": "c9", "criterion": "Clarity & Concision — clear, unambiguous writing without unnecessary verbosity.", "weight": 2},
                                {"local_id": "c10", "criterion": "Tone & Helpfulness — professional, helpful, and appropriate tone for the user's request.", "weight": 2},
                            ]
                            for pid in still_missing:
                                prompt_text = prompt_text_by_id.get(
                                    pid, str(pid)
                                )
                                current_rubrics[pid] = {
                                    "id": pid,
                                    "prompt": prompt_text,
                                    "original_rubric": "",
                                    "total_criteria": len(default_criteria),
                                    "criteria": default_criteria,
                                    "total_weight": sum(
                                        c.get("weight", 0)
                                        for c in default_criteria
                                    ),
                                }
                            log.info(
                                f"Backfilled {len(still_missing)} default rubrics"
                            )
                        self._save_initial_rubrics(current_rubrics)
                except Exception as gen_missing_err:
                    log.warning(
                        f"Failed to generate missing rubrics: {gen_missing_err}"
                    )

                self.current_rubrics = current_rubrics
            else:
                log.info("Generating initial rubrics from prompts")
                sampled = self._prepare_prompts_for_initial_rubrics(
                    current_responses
                )
                current_rubrics = self.proposer_adapter.generate_initial_rubrics(
                    sampled, self.config.models.rubric_generator
                )
                self._save_initial_rubrics(current_rubrics)

            self.current_rubrics = current_rubrics
            for _pid in self.current_rubrics:
                self.rubric_last_modified_iteration.setdefault(_pid, 0)

            # Run workflow
            if not current_responses:
                termination_reason = "insufficient_responses"
            elif self.config.workflow.max_iterations == 1:
                # Single mode: improve rubrics directly, no scoring
                termination_reason = self._run_single_mode(
                    current_responses, current_rubrics
                )
            else:
                # Iterative mode: score → select top-2 → improve → repeat
                termination_reason = self._run_iterative_mode(
                    current_responses, current_rubrics
                )

            self.final_rubrics_by_prompt = self.current_rubrics.copy()
            self._save_final_rubrics()

        finally:
            log.info("Cleaning up workflow resources")
            if hasattr(self, "proposer_adapter"):
                self.proposer_adapter.cleanup()
            if hasattr(self, "verifier_adapter"):
                self.verifier_adapter.cleanup()

        return WorkflowResult(
            status=termination_reason,
            total_time=time.time() - start_time,
            iterations=self.iterations,
            final_rubrics_by_prompt=self.final_rubrics_by_prompt,
            prompt_resolution_tracker=self.prompt_resolution_tracker,
        )

    # ------------------------------------------------------------------
    # Single mode
    # ------------------------------------------------------------------

    def _run_single_mode(
        self,
        responses: Dict[str, Dict],
        rubrics: Dict[str, Dict],
    ) -> str:
        """Improve rubrics directly from the 2 responses per prompt.

        No scoring step — with only 2 responses there is nothing to rank.
        """
        log.info("Single mode: improving rubrics directly (no scoring)")

        # Format responses so improve_rubrics can consume them.
        # No scores — both responses will be used directly.
        scored_data = []
        for pid, data in responses.items():
            rubric_obj = rubrics.get(pid, {})
            rubric_text = json.dumps(rubric_obj.get("criteria", []))
            scored_data.append({
                "id": pid,
                "prompt": data.get("prompt", str(pid)),
                "rubric": rubric_text,
                "scored_responses": [
                    {"response": r, "score": 1.0}
                    for r in data.get("responses", [])[:2]
                ],
            })

        improved = self.proposer_adapter.improve_rubrics(
            scored_responses=scored_data,
            current_rubrics=rubrics,
            model=self.config.models.rubric_generator,
        )
        self.current_rubrics = self.rubric_service.merge_rubrics(
            rubrics, improved
        )
        log.info(
            f"Single mode complete: improved {len(improved)} rubrics"
        )
        return "completed"

    # ------------------------------------------------------------------
    # Iterative mode
    # ------------------------------------------------------------------

    def _run_iterative_mode(
        self,
        current_responses: Dict[str, Dict],
        current_rubrics: Dict[str, Dict],
    ) -> str:
        """Score → select top-2 → improve rubrics → repeat."""
        termination_reason = "max_iterations_reached"

        for iteration_num in range(
            1, self.config.workflow.max_iterations + 1
        ):
            log.info(f"\n{'=' * 60}")
            log.info(f"Starting iteration {iteration_num}")
            log.info(f"{'=' * 60}")

            current_prompt_ids = list(current_responses.keys())
            if not current_prompt_ids:
                log.warning("No prompts with responses left to process.")
                termination_reason = "responses_exhausted"
                break

            log.info(
                f"Processing {len(current_prompt_ids)} prompts with responses"
            )
            (
                iter_info,
                highest_scores_by_prompt,
                updated_rubrics,
                next_responses,
            ) = self.iteration_manager.run_iteration(
                iteration_num=iteration_num,
                responses=current_responses,
                current_rubrics=current_rubrics,
                prompt_ids=current_prompt_ids,
                config={
                    "verifier_model": self.config.models.verifier,
                    "rubric_model": self.config.models.rubric_generator,
                    "responses_per_prompt": self.config.workflow.initial_responses_per_prompt,
                    "exclude_used_reference_responses": self.config.workflow.get(
                        "exclude_used_reference_responses", False
                    ),
                },
            )

            self.iterations.append(iter_info)

            for _pid, _new_rubric in updated_rubrics.items():
                if current_rubrics.get(_pid) != _new_rubric:
                    self.rubric_last_modified_iteration[_pid] = iteration_num

            resolved_prompt_ids = set(current_responses.keys()) - set(
                next_responses.keys()
            )
            for prompt_id in resolved_prompt_ids:
                if prompt_id not in self.prompt_resolution_tracker:
                    self.prompt_resolution_tracker[prompt_id] = {
                        "resolved_at_iteration": iteration_num,
                        "highest_score": highest_scores_by_prompt.get(
                            prompt_id
                        ),
                        "reason": "responses_exhausted",
                    }
                    log.info(
                        f"Prompt {prompt_id} resolved at iteration {iteration_num}"
                    )

            current_rubrics = updated_rubrics
            current_responses = next_responses

            if not current_responses:
                termination_reason = "responses_exhausted"
                break

        self.current_rubrics = current_rubrics

        # Save iterative-mode outputs
        output_dir = self.config.files.output_dir
        summary = {
            "total_iterations": len(self.iterations),
            "total_prompts_processed": len(self.current_rubrics),
            "prompts_resolved": len(self.prompt_resolution_tracker),
            "prompts_remaining": len(self.current_rubrics)
            - len(self.prompt_resolution_tracker),
            "prompt_resolution_details": self.prompt_resolution_tracker,
            "iterations": [
                {"iteration": it.iteration, "status": it.status}
                for it in self.iterations
            ],
        }
        self.file_handler.save_json(
            summary, f"{output_dir}/workflow_summary.json"
        )
        resolution_report = {
            "resolutions": self.prompt_resolution_tracker,
            "unresolved_prompt_ids": [
                pid
                for pid in self.current_rubrics
                if pid not in self.prompt_resolution_tracker
            ],
        }
        self.file_handler.save_json(
            resolution_report, f"{output_dir}/prompt_resolution_report.json"
        )

        return termination_reason

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_final_rubrics(self) -> None:
        """Save final_rubrics.json (used by both single and iterative modes)."""
        output_dir = self.config.files.output_dir
        rubrics = self.current_rubrics

        if not rubrics:
            return

        rubrics_list = []
        for pid, rubric in rubrics.items():
            clean = rubric.copy()
            clean.pop("original_rubric", None)
            clean.pop("response_selection_info", None)
            rubrics_list.append(clean)

        self.file_handler.save_json(
            rubrics_list, f"{output_dir}/final_rubrics.json"
        )
        log.info(f"Saved {len(rubrics_list)} final rubrics to {output_dir}")

    def _prepare_prompts_for_initial_rubrics(
        self, responses: Dict[str, Dict]
    ) -> List[Dict]:
        """Prepare prompt payload used for initial rubric generation."""
        return [
            {"id": prompt_id, "prompt": data.get("prompt", str(prompt_id))}
            for prompt_id, data in responses.items()
        ]

    def _save_initial_rubrics(self, rubrics: Dict[str, Dict]) -> None:
        output_dir = self.config.files.output_dir
        self.file_handler.save_json(
            list(rubrics.values()), f"{output_dir}/initial_rubrics.json"
        )
        log.info(
            f"Saved {len(rubrics)} initial rubrics to "
            f"{output_dir}/initial_rubrics.json"
        )

    def _normalize_responses(self, responses_data) -> Dict[str, Dict]:
        """Normalize various response payloads into a prompt-centric dict."""
        normalized: Dict[str, Dict] = {}

        if isinstance(responses_data, dict):
            if "responses" in responses_data and isinstance(
                responses_data["responses"], list
            ):
                responses_data = responses_data["responses"]
            else:
                responses_data = list(responses_data.values())

        if isinstance(responses_data, list) and responses_data:
            if (
                isinstance(responses_data[0], dict)
                and "responses" in responses_data[0]
            ):
                for item in responses_data:
                    prompt_id = item.get("id")
                    if prompt_id is None:
                        continue
                    normalized[prompt_id] = {
                        "id": prompt_id,
                        "prompt": item.get("prompt", str(prompt_id)),
                        "responses": self._coerce_responses(
                            item.get("responses", [])
                        ),
                    }
            else:
                for response in responses_data:
                    prompt_id = response.get(
                        "prompt_id", response.get("id")
                    )
                    if prompt_id is None:
                        continue
                    normalized.setdefault(
                        prompt_id,
                        {
                            "id": prompt_id,
                            "prompt": response.get("prompt", str(prompt_id)),
                            "responses": [],
                        },
                    )
                    normalized[prompt_id]["responses"].append(
                        response.get("response", response)
                    )
        return normalized

    def _coerce_responses(self, response_list: List) -> List[str]:
        """Ensure response values are simple strings."""
        coerced = []
        for resp in response_list:
            if isinstance(resp, dict):
                coerced.append(resp.get("response", str(resp)))
            else:
                coerced.append(resp)
        return coerced

    def _limit_initial_responses(
        self, responses: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Apply the initial_responses_per_prompt cap and prune under-populated prompts."""
        max_responses = getattr(
            self.config.workflow, "initial_responses_per_prompt", None
        )
        filtered: Dict[str, Dict] = {}
        skipped = 0
        for prompt_id, data in responses.items():
            resp_list = data.get("responses", [])
            if max_responses and max_responses > 0:
                resp_list = resp_list[:max_responses]
            if len(resp_list) < 2:
                skipped += 1
                continue
            filtered[prompt_id] = {
                "id": prompt_id,
                "prompt": data.get("prompt", str(prompt_id)),
                "responses": resp_list,
            }
        if skipped:
            log.warning(
                f"Skipped {skipped} prompts with fewer than 2 responses."
            )
        return filtered
