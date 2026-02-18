"""Workflow engine for the rubrics improvement system."""

import os
import logging
import time
from omegaconf import DictConfig, OmegaConf

from .orchestration import WorkflowCoordinator
from .data_structures import WorkflowResult
from generator.utils import set_base_url

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = "workflow_outputs"


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

        # Configure the global API base URL if set in config
        api_base_url = getattr(cfg.api, "base_url", None) if hasattr(cfg, "api") else None
        if api_base_url:
            set_base_url(str(api_base_url))

        # Determine output directory: resume if configured, otherwise create new run dir
        resume_path = getattr(self.cfg, "resume", None)
        resume_path = getattr(resume_path, "path", None) if resume_path else None
        if resume_path:
            output_dir = resume_path
            log.info(f"Resuming run from: {output_dir}")
            if not os.path.isdir(output_dir):
                raise RuntimeError(
                    f"Resume path does not exist or is not a directory: {output_dir}"
                )
            prior_cfg_path = os.path.join(output_dir, "config.yaml")
            if os.path.isfile(prior_cfg_path):
                try:
                    original_cfg = OmegaConf.load(prior_cfg_path)
                    log.info(f"Loaded original run config from {prior_cfg_path}")
                except Exception as e:
                    log.warning(
                        f"Failed to load prior config at {prior_cfg_path}: {e}"
                    )
                    original_cfg = self.cfg
            else:
                log.warning(
                    f"No prior config.yaml found at {prior_cfg_path}; "
                    "falling back to launch config for base"
                )
                original_cfg = self.cfg

            resume_overrides_path = os.path.join(output_dir, "resume.yaml")
            if os.path.isfile(resume_overrides_path):
                try:
                    resume_overrides = OmegaConf.load(resume_overrides_path)
                    log.info(
                        f"Applying resume overrides from {resume_overrides_path}"
                    )
                    effective_cfg = OmegaConf.merge(original_cfg, resume_overrides)
                except Exception as e:
                    log.warning(f"Failed to load resume overrides: {e}")
                    effective_cfg = original_cfg
            else:
                effective_cfg = original_cfg

            cfg_dict = OmegaConf.to_container(effective_cfg, resolve=True)
            cfg_dict["files"]["output_dir"] = output_dir
            self.cfg = OmegaConf.create(cfg_dict)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            configured_subdir = getattr(self.cfg.files, "output_dir", None)
            configured_subdir = (
                str(configured_subdir).strip() if configured_subdir else ""
            )
            if configured_subdir and configured_subdir.lower() != "null":
                run_subdir = os.path.normpath(configured_subdir)
                if run_subdir in ("", "."):
                    run_subdir = f"run_{timestamp}"
                elif os.path.isabs(run_subdir) or run_subdir.startswith(".."):
                    raise ValueError(
                        "files.output_dir must be a relative path under workflow_outs"
                    )
            else:
                run_subdir = f"run_{timestamp}"

            base_output_dir = os.path.abspath(DEFAULT_OUTPUT_ROOT)
            output_dir = os.path.join(base_output_dir, run_subdir)

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            cfg_dict["files"]["output_dir"] = output_dir
            self.cfg = OmegaConf.create(cfg_dict)

            os.makedirs(output_dir, exist_ok=True)
            log.info(
                f"Created output directory under {DEFAULT_OUTPUT_ROOT}: {output_dir}"
            )

            try:
                config_path = os.path.join(output_dir, "config.yaml")
                with open(config_path, "w") as f:
                    f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
                log.info(f"Saved run config to: {config_path}")
            except Exception as e:
                log.warning(f"Failed to save config to {output_dir}: {e}")

        # Initialize the workflow coordinator with updated config
        self.coordinator = WorkflowCoordinator(self.cfg)

    def run(self) -> WorkflowResult:
        """Run the complete workflow using Hydra configuration.

        Returns:
            WorkflowResult with status and iteration information
        """
        log.info(
            f"Starting workflow with output directory: {self.cfg.files.output_dir}"
        )
        self.coordinator.setup()
        result = self.coordinator.run()
        log.info(
            f"Workflow completed. Results saved in: {self.cfg.files.output_dir}"
        )
        return result
