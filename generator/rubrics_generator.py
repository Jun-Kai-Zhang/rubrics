import random
from typing import List, Optional

# Import function APIs from corresponding files
from .score_responses import score_responses
from .generate_rubrics_without_responses import generate_rubrics_without_responses
from .improve_rubrics import improve_rubrics
from .utils import (
    RANDOM_SEED, VLLM_AVAILABLE, LLM, get_gpu_count
)

class RubricsGenerator:
    
    def __init__(self):
        self.verifier_model = None
        self.vllm_instance = None
        self.proposer_model = None
        self.proposer_vllm_instance = None
        random.seed(RANDOM_SEED)
    
    def _initialize_verifier(self, model: str, tensor_parallel_size: Optional[int] = None, max_model_len: Optional[int] = None):
        """Initialize the verifier model (vLLM instance)."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        if tensor_parallel_size is None:
            tensor_parallel_size = max(1, get_gpu_count())
        
        print(f"Initializing verifier model: {model}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        
        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": "bfloat16"
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        self.vllm_instance = LLM(**llm_kwargs)
        self.verifier_model = model
        
        print(f"Verifier model initialized successfully")
    
    def _initialize_proposer(self, model: str, tensor_parallel_size: Optional[int] = None, max_model_len: Optional[int] = None, enforce_eager: bool = False):
        """Initialize the proposer model (vLLM instance for rubric generation/improvement)."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
        
        if tensor_parallel_size is None:
            tensor_parallel_size = max(1, get_gpu_count())
        
        print(f"Initializing proposer model: {model}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        
        llm_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": "bfloat16",
            "enforce_eager": enforce_eager,
            "trust_remote_code": True,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        self.proposer_vllm_instance = LLM(**llm_kwargs)
        self.proposer_model = model
        
        print(f"Proposer model initialized successfully")
    
    def cleanup(self):
        """Clean up the vLLM instance and associated resources."""
        if hasattr(self, 'vllm_instance') and self.vllm_instance is not None:
            print("Cleaning up vLLM instance...")
            # Clean up the instance
            del self.vllm_instance
            self.vllm_instance = None
            self.verifier_model = None
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        if hasattr(self, 'proposer_vllm_instance') and self.proposer_vllm_instance is not None:
            print("Cleaning up proposer vLLM instance...")
            del self.proposer_vllm_instance
            self.proposer_vllm_instance = None
            self.proposer_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def _score_responses(self, responses_file: str, output_file: str, sample_size: int, 
                        rubrics_file: str = None, backend: str = "vllm", 
                        max_retries: int = 2, retry_temperature: float = 1.0):
        """Score responses using rubrics."""
        if rubrics_file is None:
            raise ValueError("rubrics_file is required for scoring responses")
        
        return score_responses(
            responses_file=responses_file,
            rubrics_file=rubrics_file,
            output_file=output_file,
            sample_size=sample_size,
            backend=backend,
            verifier=self.verifier_model,
            vllm_instance=self.vllm_instance,
            max_retries=max_retries,
            retry_temperature=retry_temperature
        )

    def _generate_rubrics_without_responses(self, prompts_file: str, output_file: str, sample_size: int,
                                          rubric_generator: str = "gemini/gemini-2.5-pro",
                                          prompt_template_file: str = "generator/prompts/generate_rubrics_without_responses.txt",
                                          max_workers: int = 64,
                                          backend: str = "api",
                                          vllm_tensor_parallel_size: Optional[int] = None,
                                          vllm_max_model_len: Optional[int] = None,
                                          vllm_enforce_eager: bool = False,
                                          max_retries: int = 2):
        """Generate rubrics from prompts only (without responses)."""
        # Ensure proposer vLLM instance if needed
        vllm_instance = None
        if backend == "vllm":
            if self.proposer_vllm_instance is None or self.proposer_model != rubric_generator:
                self._initialize_proposer(
                    model=rubric_generator,
                    tensor_parallel_size=vllm_tensor_parallel_size,
                    max_model_len=vllm_max_model_len,
                    enforce_eager=vllm_enforce_eager,
                )
            vllm_instance = self.proposer_vllm_instance
        return generate_rubrics_without_responses(
            prompts_file=prompts_file,
            output_file=output_file,
            sample_size=sample_size,
            rubric_generator=rubric_generator,
            prompt_template_file=prompt_template_file,
            max_workers=max_workers,
            backend=backend,
            vllm_model=rubric_generator,
            vllm_instance=vllm_instance,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_max_model_len=vllm_max_model_len,
            vllm_enforce_eager=vllm_enforce_eager,
            max_retries=max_retries,
        )

    def _improve_rubrics(self, scored_file: str, output_file: str, sample_size: int,
                        rubric_improver_model: str = "gemini/gemini-2.5-pro",
                        prompt_template_file: str = "generator/prompts/improve_rubrics.txt",
                        previous_rubrics_file: str = None,
                        selection_strategy: str = "ties",
                        backend: str = "api",
                        vllm_tensor_parallel_size: Optional[int] = None,
                        vllm_max_model_len: Optional[int] = None,
                        vllm_enforce_eager: bool = False):
        """Improve rubrics using the highest scoring responses.

        Always uses "improve" mode for full rubric replacement.
        """
        # Ensure proposer vLLM instance if needed
        vllm_instance = None
        if backend == "vllm":
            if self.proposer_vllm_instance is None or self.proposer_model != rubric_improver_model:
                self._initialize_proposer(
                    model=rubric_improver_model,
                    tensor_parallel_size=vllm_tensor_parallel_size,
                    max_model_len=vllm_max_model_len,
                    enforce_eager=vllm_enforce_eager,
                )
            vllm_instance = self.proposer_vllm_instance

        return improve_rubrics(
            scored_file=scored_file,
            output_file=output_file,
            sample_size=sample_size,
            rubric_improver_model=rubric_improver_model,
            prompt_template_file=prompt_template_file,
            previous_rubrics_file=previous_rubrics_file,
            selection_strategy=selection_strategy,
            backend=backend,
            vllm_model=rubric_improver_model,
            vllm_instance=vllm_instance,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_max_model_len=vllm_max_model_len,
            vllm_enforce_eager=vllm_enforce_eager,
        )

            
