"""File I/O operations handler."""

import json
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

log = logging.getLogger(__name__)


class FileHandler:
    """Handles file I/O operations."""
    
    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
    
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON data from file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        log.debug(f"Loading JSON from {filepath}")
        with open(filepath, 'r', encoding=self.encoding) as f:
            return json.load(f)
    
    def save_json(
        self,
        data: Any,
        filepath: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """Save data to JSON file.
        
        Args:
            data: Data to save
            filepath: Path to save file
            indent: JSON indentation level
            ensure_ascii: Whether to escape non-ASCII characters
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        log.debug(f"Saving JSON to {filepath}")
        with open(filepath, 'w', encoding=self.encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    def exists(self, filepath: str) -> bool:
        """Check if file exists."""
        return Path(filepath).exists()
    
    def create_directory(self, dirpath: str) -> None:
        """Create directory if it doesn't exist."""
        Path(dirpath).mkdir(parents=True, exist_ok=True)
    
    def list_files(
        self,
        directory: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """List files in directory matching pattern.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        dirpath = Path(directory)
        if not dirpath.exists():
            return []
        
        if recursive:
            return list(dirpath.rglob(pattern))
        return list(dirpath.glob(pattern))
    
    def delete_file(self, filepath: str) -> bool:
        """Delete file if it exists.
        
        Args:
            filepath: Path to file to delete
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            log.debug(f"Deleted {filepath}")
            return True
        return False
    
    def get_file_size(self, filepath: str) -> Optional[int]:
        """Get file size in bytes."""
        filepath = Path(filepath)
        if filepath.exists():
            return filepath.stat().st_size
        return None 