#!/usr/bin/env python3
"""Hydra-based workflow for iterative rubrics improvement."""

import sys
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from .workflow_engine import WorkflowEngine

log = logging.getLogger(__name__)

# Suppress httpx INFO logs (which come from litellm)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="./configs", config_name="single")
def main(cfg: DictConfig) -> None:
    """Main workflow execution with Hydra configuration."""
    log.info("Starting iterative rubrics improvement workflow")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    sample_desc = (
        f"{cfg.workflow.sample_size} prompts"
        if cfg.workflow.sample_size
        else "all prompts"
    )
    proposer_backend = getattr(cfg.models, "proposer_backend", "api")
    verifier_backend = getattr(cfg.models, "verifier_backend", "api")
    log.info(
        f"Config: {sample_desc}, proposer={proposer_backend}, "
        f"verifier={verifier_backend}, max {cfg.workflow.max_iterations} iterations"
    )

    engine = WorkflowEngine(cfg)
    result = engine.run()

    log.info(f"Workflow finished with status: {result.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
