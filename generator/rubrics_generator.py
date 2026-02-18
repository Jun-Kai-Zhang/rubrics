"""Backward-compatible re-exports of generator functions.

The ``RubricsGenerator`` class that previously lived here only existed to
manage in-process vLLM instances.  Now that all model calls go through an
OpenAI-compatible HTTP API, the class is no longer needed.  We keep this
module around so that any old ``from generator.rubrics_generator import …``
imports still resolve.
"""

from .score_responses import score_responses
from .generate_rubrics import generate_rubrics
from .improve_rubrics import improve_rubrics

__all__ = [
    "score_responses",
    "generate_rubrics",
    "improve_rubrics",
]
