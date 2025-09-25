"""Workflow package for the iterative rubric-generation pipeline.

Public API
==========
The higher-level CLI (or other Python code) should import only the
objects re-exported here:

• :class:`WorkflowEngine` – orchestrates the multi-iteration pipeline
• :data:`__version__` – package version

# Implementation note
All sub-modules live in the same directory, so they are imported lazily
to avoid the cost of heavy deps (omegaconf, hydra, etc.) unless they are
actually needed.
"""

from importlib import import_module

__all__ = [
    "WorkflowEngine",
    "__version__",
]

__version__ = "1.1.0"
__author__ = "Rubrics Improvement System"


# Lazy loading ----------------------------------------------------------------

def _lazy_import(name: str):
    return import_module(f"workflow.{name}")


WorkflowEngine = _lazy_import("workflow_engine").WorkflowEngine 