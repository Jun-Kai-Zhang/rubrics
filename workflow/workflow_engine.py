"""Workflow engine for the rubrics improvement system."""

import os
import logging
import time
from omegaconf import DictConfig, OmegaConf

from .orchestration import WorkflowCoordinator
from .data_structures import WorkflowResult

# For backward compatibility with generator package
from generator.utils import get_gpu_count, VLLM_AVAILABLE

log = logging.getLogger(__name__)


class WorkflowEngine:
    """Workflow engine that delegates to the modular orchestration system.
    
    This class maintains backward compatibility while using the new
    modular architecture under the hood.
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize the workflow engine with Hydra config.
        
        Args:
            cfg: Hydra structured configuration object
        """
        self.cfg = cfg
        
        # Setup vLLM if needed
        self._setup_vllm()
        
        # Determine output directory: resume if configured, otherwise create new run dir
        resume_path = getattr(self.cfg, 'resume', None)
        resume_path = getattr(resume_path, 'path', None) if resume_path else None
        if resume_path:
            # Use the existing run directory and reconstruct config from saved files
            output_dir = resume_path
            log.info(f"Resuming run from: {output_dir}")
            if not os.path.isdir(output_dir):
                raise RuntimeError(f"Resume path does not exist or is not a directory: {output_dir}")
            # Load original run config if present
            prior_cfg_path = os.path.join(output_dir, "config.yaml")
            effective_cfg: DictConfig
            if os.path.isfile(prior_cfg_path):
                try:
                    original_cfg = OmegaConf.load(prior_cfg_path)
                    log.info(f"Loaded original run config from {prior_cfg_path}")
                except Exception as e:
                    log.warning(f"Failed to load prior config at {prior_cfg_path}: {e}")
                    original_cfg = self.cfg
            else:
                log.warning(f"No prior config.yaml found at {prior_cfg_path}; falling back to launch config for base")
                original_cfg = self.cfg
            # Load optional resume-only overrides from resume.yaml in the run directory
            resume_overrides_path = os.path.join(output_dir, "resume.yaml")
            if os.path.isfile(resume_overrides_path):
                try:
                    resume_overrides = OmegaConf.load(resume_overrides_path)
                    log.info(f"Applying resume overrides from {resume_overrides_path}")
                    effective_cfg = OmegaConf.merge(original_cfg, resume_overrides)
                except Exception as e:
                    log.warning(f"Failed to load resume overrides: {e}")
                    effective_cfg = original_cfg
            else:
                effective_cfg = original_cfg
            # Ensure output_dir is exactly the resume path
            cfg_dict = OmegaConf.to_container(effective_cfg, resolve=True)
            cfg_dict['files']['output_dir'] = output_dir
            self.cfg = OmegaConf.create(cfg_dict)
        else:
            # Create timestamped output directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_output_dir = self.cfg.files.output_dir
            output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
            
            # Update the config to use the timestamped directory
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            cfg_dict['files']['output_dir'] = output_dir
            self.cfg = OmegaConf.create(cfg_dict)
            
            # Create the timestamped output directory
            os.makedirs(output_dir, exist_ok=True)
            log.info(f"Created timestamped output directory: {output_dir}")
            
            # Save a copy of the resolved config in the output directory
            try:
                config_path = os.path.join(output_dir, "config.yaml")
                with open(config_path, "w") as f:
                    f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
                log.info(f"Saved run config to: {config_path}")
            except Exception as e:
                log.warning(f"Failed to save config to {output_dir}: {e}")
        
        # Initialize the new workflow coordinator with updated config
        self.coordinator = WorkflowCoordinator(self.cfg)
    
    def _setup_vllm(self) -> None:
        """Removed vLLM support - always use API."""
        pass
    
    def run(self) -> WorkflowResult:
        """Run the complete workflow using Hydra configuration.
        
        Returns:
            WorkflowResult with status and iteration information
        """
        log.info(f"Starting workflow with output directory: {self.cfg.files.output_dir}")
        
        # Setup the coordinator
        self.coordinator.setup()
        
        # Delegate to the coordinator
        result = self.coordinator.run()
        
        # Log final output location
        log.info(f"Workflow completed. Results saved in: {self.cfg.files.output_dir}")
        
        return result 