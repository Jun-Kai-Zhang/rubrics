"""Adapter for external model integrations."""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from workflow.core.interfaces import RubricGenerator, ResponseScorer

log = logging.getLogger(__name__)


class ModelAdapter(RubricGenerator, ResponseScorer):
    """Adapter for external model operations.
    
    This adapter wraps the external generator package to implement
    our domain interfaces.
    """
    
    def __init__(self):
        """Initialize model adapter."""
        self._generator = None
        self._initialized = False
    
    def _ensure_generator(self) -> None:
        """Lazily initialize the generator."""
        if not self._initialized:
            # Import here to avoid circular dependencies
            from generator.rubrics_generator import RubricsGenerator
            self._generator = RubricsGenerator()
            self._initialized = True
    
    def generate_initial_rubrics(
        self,
        responses: List[Dict],
        model: str
    ) -> Dict[str, Dict]:
        """Generate initial rubrics from responses.
        
        Args:
            responses: List of response data
            model: Model name to use for generation
            
        Returns:
            Dictionary mapping prompt_id to rubric
        """
        self._ensure_generator()
        
        # Create temporary file for responses
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(responses, f)
            temp_responses_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output_file = f.name
        
        try:
            # Call generator method
            print("Generating rubrics without responses...")
            # Choose prompt template for generation based on adapter flag
            prompt_template_file = "generator/prompts/generate_rubrics_without_responses_simplified.txt"

            success = self._generator._generate_rubrics_without_responses(
                prompts_file=temp_responses_file,
                output_file=temp_output_file,
                sample_size=None,
                rubric_generator=model,
                prompt_template_file=prompt_template_file,
            )
            
            if not success:
                raise RuntimeError("Failed to generate initial rubrics")
            
            # Load and parse results
            with open(temp_output_file, 'r') as f:
                data = json.load(f)
            
            # Convert to dict format
            rubrics_dict = {}
            for rubric in data.get("rubrics", []):
                rubrics_dict[rubric["id"]] = rubric
            
            return rubrics_dict
            
        finally:
            # Cleanup temp files
            Path(temp_responses_file).unlink(missing_ok=True)
            Path(temp_output_file).unlink(missing_ok=True)
    
    def improve_rubrics(
        self,
        scored_responses: Dict,
        current_rubrics: Dict[str, Dict],
        model: str
    ) -> Dict[str, Dict]:
        """Improve rubrics based on scoring results.
        
        Args:
            scored_responses: Scored response data
            current_rubrics: Current rubrics
            model: Model name to use for improvement
            
        Returns:
            Dictionary mapping prompt_id to improved rubric
        """
        self._ensure_generator()
        
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scored_responses, f)
            temp_scored_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output_file = f.name
        
        # Save current rubrics to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Format current rubrics in the expected structure
            rubrics_data = {
                "rubrics": list(current_rubrics.values())
            }
            json.dump(rubrics_data, f)
            temp_rubrics_file = f.name
        
        try:
            # Use the simplified prompt template
            prompt_template_file = "generator/prompts/improve_rubrics_simplified.txt"

            # Call generator method with fixed values
            self._generator._improve_rubrics(
                scored_file=temp_scored_file,
                output_file=temp_output_file,
                sample_size=None,
                rubric_improver_model=model,
                prompt_template_file=(prompt_template_file if prompt_template_file else "generator/prompts/improve_rubrics_simplified.txt"),
                force_continue=True,  # Always true
                previous_rubrics_file=temp_rubrics_file,
                selection_strategy="top2",  # Always top2
            )
            
            # Load and parse results
            with open(temp_output_file, 'r') as f:
                data = json.load(f)
            
            # Convert to dict format
            improved_dict = {}
            for rubric in data.get("rubrics", []):
                improved_dict[rubric["id"]] = rubric
            
            return improved_dict
            
        finally:
            # Cleanup temp files
            Path(temp_scored_file).unlink(missing_ok=True)
            Path(temp_output_file).unlink(missing_ok=True)
            Path(temp_rubrics_file).unlink(missing_ok=True)
    
    def score_responses(
        self,
        responses: List[Dict],
        rubrics: Dict[str, Dict],
        model: str
    ) -> Dict:
        """Score responses using rubrics.
        
        Args:
            responses: List of response data
            rubrics: Dictionary of rubrics by prompt_id
            model: Model name to use for scoring
            
        Returns:
            Scored response data
        """
        self._ensure_generator()
        
        import tempfile
        import json
        
        # Always use API backend, just set the verifier model
        self._generator.verifier_model = model
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(responses, f)
            temp_responses_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Convert rubrics dict to list format
            rubrics_list = list(rubrics.values())
            json.dump({"rubrics": rubrics_list}, f)
            temp_rubrics_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_output_file = f.name
        
        try:
            # Call generator method
            self._generator._score_responses(
                responses_file=temp_responses_file,
                output_file=temp_output_file,
                sample_size=None,
                rubrics_file=temp_rubrics_file,
                max_retries=2,
                retry_temperature=1.0
            )
            
            # Load and return results
            with open(temp_output_file, 'r') as f:
                return json.load(f)
            
        finally:
            # Cleanup temp files
            Path(temp_responses_file).unlink(missing_ok=True)
            Path(temp_rubrics_file).unlink(missing_ok=True)
            Path(temp_output_file).unlink(missing_ok=True) 

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._initialized and self._generator is not None:
            log.info("Cleaning up model adapter resources")
            if hasattr(self._generator, 'cleanup'):
                self._generator.cleanup()
            self._generator = None
            self._initialized = False 