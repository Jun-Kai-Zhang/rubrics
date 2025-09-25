"""Top-level package for rubric generation helpers.

This file makes ``generator`` a real Python package so that other
modules (e.g. the future ``workflow`` package) can simply
``import generator`` or ``from generator import X`` without relying on
manual ``sys.path`` manipulation inside each script.

It intentionally keeps a very small public surface – only the high-level
``RubricsGenerator`` orchestrator.  Down-stream code should import other
modules explicitly (``from generator.utils import …``).
"""

from importlib import import_module

__all__ = [
    "RubricsGenerator",
]

# Lazily import the orchestrator to avoid importing heavy deps (vLLM etc.)
RubricsGenerator = import_module("generator.rubrics_generator").RubricsGenerator 