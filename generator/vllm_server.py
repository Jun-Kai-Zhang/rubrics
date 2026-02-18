"""VLLMServer — manage a vLLM OpenAI-compatible server as a subprocess.

When the workflow config sets ``backend: "vllm"``, the :class:`ModelAdapter`
creates a :class:`VLLMServer`, starts it, and routes all ``generate_via_api``
calls through its ``base_url``.  This keeps the generator code unified: every
model call goes through the same OpenAI-compatible HTTP interface regardless
of whether the model is cloud-hosted or running locally via vLLM.
"""

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)


class VLLMServer:
    """Wraps ``vllm.entrypoints.openai.api_server`` as a managed subprocess."""

    def __init__(
        self,
        model: str,
        port: int = 8000,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model = model
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size or 1
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager
        self.gpu_memory_utilization = gpu_memory_utilization

        self.base_url = f"http://localhost:{port}/v1"
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: int = 300) -> "VLLMServer":
        """Launch the vLLM server and block until it is ready.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for the server to become healthy.

        Returns
        -------
        self, for convenient chaining.
        """
        cmd = self._build_command()
        log.info("Starting vLLM server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # Safety net: stop the server when the Python process exits.
        atexit.register(self.stop)

        self._wait_for_ready(timeout)
        log.info(
            "vLLM server ready — model=%s  port=%d  url=%s",
            self.model,
            self.port,
            self.base_url,
        )
        return self

    def stop(self) -> None:
        """Terminate the server process (if running)."""
        if self._process is None or self._process.poll() is not None:
            return
        log.info("Stopping vLLM server (pid=%d) …", self._process.pid)
        # Kill the whole process group so child workers also exit.
        os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.warning("vLLM server did not exit in 30 s — sending SIGKILL")
            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            self._process.wait(timeout=10)
        self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_command(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        if self.max_model_len is not None:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd

    def _wait_for_ready(self, timeout: int) -> None:
        """Poll ``/v1/models`` until the server responds with 200."""
        url = f"http://localhost:{self.port}/v1/models"
        deadline = time.monotonic() + timeout
        poll_interval = 2  # seconds

        while time.monotonic() < deadline:
            # Check that the process hasn't crashed.
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited with code {self._process.returncode} "
                    f"before becoming ready"
                )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(poll_interval)

        # Timed out — kill the server and raise.
        self.stop()
        raise TimeoutError(
            f"vLLM server did not become ready within {timeout} seconds"
        )
