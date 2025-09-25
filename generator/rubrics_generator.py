import os
import json
import random
from typing import Dict, List, Optional, Any

# Import function APIs from corresponding files
from .score_responses import score_responses
from .generate_rubrics_without_responses import generate_rubrics_without_responses
from .improve_rubrics import improve_rubrics
from .utils import (
    RANDOM_SEED
)

class RubricsGenerator:
    
    def __init__(self):
        self.verifier_model = None
        self.proposer_model = None
        random.seed(RANDOM_SEED)
    
    # Removed vLLM support - always use API
    
    
    def cleanup(self):
        """Clean up resources."""
        pass


    def _score_responses(self, responses_file: str, output_file: str, sample_size: int, 
                        rubrics_file: str = None,
                        max_retries: int = 2, retry_temperature: float = 1.0):
        """Score responses using rubrics."""
        if rubrics_file is None:
            raise ValueError("rubrics_file is required for scoring responses")
        
        return score_responses(
            responses_file=responses_file,
            rubrics_file=rubrics_file,
            output_file=output_file,
            sample_size=sample_size,
            verifier=self.verifier_model,
            max_retries=max_retries,
            retry_temperature=retry_temperature
        )


    def _generate_rubrics_without_responses(self, prompts_file: str, output_file: str, sample_size: int,
                                          rubric_generator: str = "gemini/gemini-2.5-pro",
                                          prompt_template_file: str = "generator/prompts/generate_rubrics_without_responses_simplified.txt",
                                          max_workers: int = 64):
        """Generate rubrics from prompts only (without responses)."""
        return generate_rubrics_without_responses(
            prompts_file=prompts_file,
            output_file=output_file,
            sample_size=sample_size,
            rubric_generator=rubric_generator,
            prompt_template_file=prompt_template_file,
            max_workers=max_workers
        )

    def _improve_rubrics(self, scored_file: str, output_file: str, sample_size: int,
                        rubric_improver_model: str = "gemini/gemini-2.5-pro",
                        prompt_template_file: str = "generator/prompts/improve_rubrics_simplified.txt",
                        previous_rubrics_file: str = None,
                        # Fixed values
                        force_continue: bool = True,
                        selection_strategy: str = "top2"):
        """Improve rubrics using the highest scoring responses."""
        return improve_rubrics(
            scored_file=scored_file,
            output_file=output_file,
            sample_size=sample_size,
            rubric_improver_model=rubric_improver_model,
            prompt_template_file=prompt_template_file,
            force_continue=force_continue,
            previous_rubrics_file=previous_rubrics_file,
            selection_strategy=selection_strategy
        )

            

