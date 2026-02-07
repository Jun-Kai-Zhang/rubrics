#!/usr/bin/env python3
"""Hydra-based workflow for iterative rubrics improvement."""

import sys
import os
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

# Relative import of engine within package
from .workflow_engine import WorkflowEngine

log = logging.getLogger(__name__)

# Suppress httpx INFO logs (which come from litellm)
logging.getLogger("httpx").setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="./configs", config_name="single")
def main(cfg: DictConfig) -> None:
    """Main workflow execution with Hydra configuration.
    
    Args:
        cfg: Hydra configuration loaded from YAML
    """
    # Set vLLM environment variable
    # os.environ["VLLM_USE_V1"] = "0"
    
    # Log configuration
    log.info("Starting iterative rubrics improvement workflow")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    # Print workflow summary
    sample_desc = f"{cfg.workflow.sample_size} prompts" if cfg.workflow.sample_size else "all prompts"
    proposer_backend = getattr(cfg.models, 'proposer_backend', 'api')
    verifier_backend = getattr(cfg.models, 'verifier_backend', 'api')
    log.info(f"Config: {sample_desc}, proposer={proposer_backend}, verifier={verifier_backend}, max {cfg.workflow.max_iterations} iterations")
    
    # Create and run workflow engine
    engine = WorkflowEngine(cfg)
    
    try:
        result = engine.run()
        
        # Exit with appropriate code
        if result.status == "success_no_ties":
            log.info("Workflow completed successfully!")
            return 0
        else:
            log.warning(f"Workflow ended with status: {result.status}")
            return 1
            
    except Exception as e:
        log.error(f"Workflow failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 