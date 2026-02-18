"""Adapter for external model integrations.

Routes all model calls through an OpenAI-compatible HTTP API.  When
``backend="vllm"``, a :class:`VLLMServer` is launched automatically and its
URL is used as ``base_url``.  When ``backend="api"`` the configured litellm
proxy URL (or default) is used instead.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from workflow.core.interfaces import RubricGenerator, ResponseScorer

log = logging.getLogger(__name__)


class ModelAdapter(RubricGenerator, ResponseScorer):
    """Adapter that delegates to generator functions via HTTP API.

    Parameters
    ----------
    backend:
        ``"api"`` to use a litellm proxy, ``"vllm"`` to auto-start a local
        vLLM OpenAI-compatible server.
    vllm_config:
        Configuration forwarded to :class:`VLLMServer` when *backend* is
        ``"vllm"``.  Keys: ``tensor_parallel_size``, ``max_model_len``,
        ``enforce_eager``, ``gpu_memory_utilization``, ``port``.
    api_config:
        API-related settings.  ``base_url`` overrides the default litellm
        URL; ``workers``, ``max_retries``, ``retry_temperature`` control
        request behaviour.
    """

    def __init__(
        self,
        backend: str = "api",
        vllm_config: Optional[Dict] = None,
        api_config: Optional[Dict] = None,
    ):
        self.backend = backend
        self.vllm_config = vllm_config or {}
        self.api_config = api_config or {}

        self._vllm_server = None  # VLLMServer instance (if any)
        self._base_url: Optional[str] = self.api_config.get("base_url")

    # ------------------------------------------------------------------
    # vLLM server lifecycle
    # ------------------------------------------------------------------

    def _ensure_vllm_server(self, model: str) -> None:
        """Start (or restart) the vLLM server for *model* if needed."""
        if self._vllm_server is not None and self._vllm_server.model == model:
            return  # already running with the right model

        # Stop any existing server first
        if self._vllm_server is not None:
            self._vllm_server.stop()
            self._vllm_server = None

        from generator.vllm_server import VLLMServer

        server = VLLMServer(
            model=model,
            port=self.vllm_config.get("port", 8000),
            tensor_parallel_size=self.vllm_config.get("tensor_parallel_size", 1),
            max_model_len=self.vllm_config.get("max_model_len"),
            enforce_eager=self.vllm_config.get("enforce_eager", False),
            gpu_memory_utilization=self.vllm_config.get(
                "gpu_memory_utilization", 0.9
            ),
        )
        server.start()
        self._vllm_server = server
        self._base_url = server.base_url

    def _get_base_url(self, model: str) -> Optional[str]:
        """Return the API base URL, starting a vLLM server if needed."""
        if self.backend == "vllm":
            self._ensure_vllm_server(model)
        return self._base_url

    # ------------------------------------------------------------------
    # RubricGenerator interface
    # ------------------------------------------------------------------

    def generate_initial_rubrics(
        self,
        responses: List[Dict],
        model: str,
    ) -> Dict[str, Dict]:
        """Generate initial rubrics from prompts.

        Returns a dict mapping ``prompt_id → rubric``.
        """
        from generator.generate_rubrics import generate_rubrics

        base_url = self._get_base_url(model)
        worker_count = self.api_config.get("workers") or 64
        prompt_template_file = (
            "generator/prompts/generate_rubrics.txt"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(responses, f)
            temp_prompts = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_output = f.name

        try:
            print("Generating rubrics...")
            success = generate_rubrics(
                prompts_file=temp_prompts,
                output_file=temp_output,
                sample_size=None,
                rubric_generator=model,
                prompt_template_file=prompt_template_file,
                max_workers=worker_count,
                max_retries=self.api_config.get("max_retries", 2),
                base_url=base_url,
            )
            if not success:
                raise RuntimeError("Failed to generate initial rubrics")

            with open(temp_output, "r") as f:
                data = json.load(f)

            return {r["id"]: r for r in data}
        finally:
            Path(temp_prompts).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)

    def improve_rubrics(
        self,
        scored_responses: Dict,
        current_rubrics: Dict[str, Dict],
        model: str,
    ) -> Dict[str, Dict]:
        """Improve rubrics based on scoring results.

        Returns a dict mapping ``prompt_id → improved rubric``.
        """
        from generator.improve_rubrics import improve_rubrics

        base_url = self._get_base_url(model)
        prompt_template_file = "generator/prompts/improve_rubrics.txt"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(scored_responses, f)
            temp_scored = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_output = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(list(current_rubrics.values()), f)
            temp_rubrics = f.name

        try:
            improve_rubrics(
                scored_file=temp_scored,
                output_file=temp_output,
                sample_size=None,
                rubric_improver_model=model,
                prompt_template_file=prompt_template_file,
                previous_rubrics_file=temp_rubrics,
                base_url=base_url,
            )

            with open(temp_output, "r") as f:
                data = json.load(f)

            return {r["id"]: r for r in data}
        finally:
            Path(temp_scored).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)
            Path(temp_rubrics).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # ResponseScorer interface
    # ------------------------------------------------------------------

    def score_responses(
        self,
        responses: List[Dict],
        rubrics: Dict[str, Dict],
        model: str,
    ) -> Dict:
        """Score responses using rubrics.

        Returns the scored response data structure.
        """
        from generator.score_responses import score_responses

        base_url = self._get_base_url(model)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(responses, f)
            temp_responses = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(list(rubrics.values()), f)
            temp_rubrics = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_output = f.name

        try:
            score_responses(
                responses_file=temp_responses,
                rubrics_file=temp_rubrics,
                output_file=temp_output,
                sample_size=None,
                verifier=model,
                max_retries=self.api_config.get("max_retries", 2),
                retry_temperature=self.api_config.get("retry_temperature", 1.0),
                base_url=base_url,
            )

            with open(temp_output, "r") as f:
                return json.load(f)
        finally:
            Path(temp_responses).unlink(missing_ok=True)
            Path(temp_rubrics).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop the vLLM server (if one was started)."""
        if self._vllm_server is not None:
            log.info("Stopping vLLM server")
            self._vllm_server.stop()
            self._vllm_server = None
