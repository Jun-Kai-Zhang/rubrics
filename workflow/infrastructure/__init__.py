"""Infrastructure layer for external dependencies and I/O operations."""

from .file_handler import FileHandler
from .model_adapter import ModelAdapter

__all__ = [
    "FileHandler",
    "ModelAdapter"
] 