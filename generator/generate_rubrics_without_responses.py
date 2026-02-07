#!/usr/bin/env python3
"""generate_rubrics_without_responses.py

Generate rubrics from prompts only (without responses) using GPT models.

"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# vLLM imports (optional)
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# ---------------------------------------------------------------------------
# Local imports – prefer package-relative, with fallback for direct execution.
# ---------------------------------------------------------------------------
import sys
try:
    from .utils import generate_via_api  # type: ignore
except ImportError:  # pragma: no cover
    print("ImportError: utils.py not found, using path hack")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import generate_via_api  # type: ignore

# ---------------------------------------------------------------------------
# Configuration defaults -----------------------------------------------------
# ---------------------------------------------------------------------------
RUBRIC_GENERATOR: str = "gemini/gemini-2.5-pro"
MAX_TOKENS: Optional[int] = None
TEMPERATURE: float = 0.0
NUM_WORKERS: int = 64
RANDOM_SEED: int = 42

# File paths
DEFAULT_DATA_FILE = "data/exp0.3/Policy_Model_Qwen2.5_32B_Instruct_Temperature_1.0_TopP_0.95_1000_Prompts_64_Tesponses_Dataset_OST.json"
PROMPT_TEMPLATE_FILE = "generator/prompts/generate_rubrics_without_responses.txt"


def generate_default_output_filename(input_file: str, generator: str, sample_size: Optional[int] = None) -> str:
    """Generate a default output filename based on input file and generator."""
    import os
    
    # Extract base name from input file (without extension)
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    
    # Clean up generator name for filename (replace slashes with underscores)
    generator_clean = generator.split("/")[-1].replace("/", "_").replace("-", "_")
    
    # Get directory from input file to place output in the same directory
    input_dir = os.path.dirname(input_file)
    
    # Build filename components
    filename_parts = ["Rubrics_No_Responses", "Generator_" + generator_clean, "Input_File_" + input_base]
    
    # Add sample size if specified
    if sample_size and sample_size > 0:
        filename_parts.append(f"Sample_{sample_size}")
    
    # Create filename
    output_filename = "_".join(filename_parts) + ".json"
    
    # Combine with directory
    return os.path.join("data/exp2", output_filename)


# ---------------------------------------------------------------------------
# Argument parsing -----------------------------------------------------------
# ---------------------------------------------------------------------------
def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate rubrics from prompts only (without responses).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="Input JSON file with prompts data (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=RUBRIC_GENERATOR,
        help="Model to use for rubric generation (HF id for vLLM, provider/model for API)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of parallel worker threads to use (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of prompts to sample from the dataset (default: process all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on input and generator)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["api", "vllm"],
        default="api",
        help="Backend to use: 'api' for OpenAI-compatible HTTP API, 'vllm' for local vLLM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS if MAX_TOKENS is not None else 0,
        help="Max new tokens to generate (0 to use default)",
    )
    # vLLM-specific options
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size (GPUs). If not set, auto-detect",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="vLLM max model length",
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action="store_true",
        help="Enable vLLM eager mode",
    )
    return parser


def load_prompt_template(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rubric_prompt(prompt: str, template: str) -> List[Dict]:
    """Return conversation that asks a model to construct a rubric for *prompt* without reference responses."""
    # Replace placeholders in the template
    user_msg = template.format(prompt=prompt)
    
    return [{"role": "user", "content": user_msg}], user_msg


def fix_json_with_gemini_flash(malformed_json: str) -> str:
    """Try to fix malformed JSON using gemini-2.5-flash."""
    fix_prompt = f"""The following text should be a valid JSON but has formatting issues. Please fix it and return only the valid JSON:

{malformed_json}

Return only the corrected JSON, no explanations or markdown formatting."""
    
    conv = [{"role": "user", "content": fix_prompt}]
    
    try:
        fixed_response = generate_via_api(
            "gemini/gemini-2.5-flash",
            conv,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        
        # Clean up the response
        response_text = fixed_response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        return response_text
    except Exception as e:
        print(f"Error using gemini-2.5-flash to fix JSON: {e}")
        return malformed_json


def generate_rubric(example: Dict, template: str) -> Dict:
    """Generate a rubric for a given example."""
    prompt = example["prompt"]
    
    conv, user_msg = build_rubric_prompt(prompt, template)
    
    # Call the API with internal retries; if it still fails, return an error entry for this prompt
    try:
        rubric_text = generate_via_api(
            RUBRIC_GENERATOR,
            conv,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as api_error:
        return {
            "id": example.get("id"),
            "prompt": example.get("prompt"),
            "original_rubric": "",
            "error": f"API error: {api_error}",
            "total_criteria": 0,
            "criteria": [],
            "total_weight": 0,
        }
    
    # Parse the JSON response
    try:
        # Clean up the response
        response_text = rubric_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        # Parse JSON
        rubric_criteria = json.loads(response_text)
        
        # Convert to the format expected by score_responses.py
        criteria_list = []
        if isinstance(rubric_criteria, list):
            criteria_list = rubric_criteria
        else:
            raise ValueError(f"Invalid rubric format: {type(rubric_criteria)}")
        # elif isinstance(rubric_criteria, dict):
        #     # Convert dict format to list format
        #     for key, value in rubric_criteria.items():
        #         if isinstance(value, dict):
        #             criterion = value.copy()
        #             if "criterion" not in criterion:
        #                 criterion["criterion"] = key
        #             criteria_list.append(criterion)
        #         else:
        #             criteria_list.append({
        #                 "criterion": key,
        #                 "weight": value if isinstance(value, (int, float)) else 1,
        #                 "description": str(value) if not isinstance(value, (int, float)) else ""
        #             })
        
        # Add local_id to each criterion and ensure weights are numeric
        for i, criterion in enumerate(criteria_list):
            if "local_id" not in criterion:
                criterion["local_id"] = f"c{i+1}"
            # Ensure weight is numeric
            if "weight" not in criterion or not isinstance(criterion.get("weight"), (int, float)):
                criterion["weight"] = 1
        
        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "original_rubric": rubric_text.strip(),
            "total_criteria": len(criteria_list),
            "criteria": criteria_list,
            "total_weight": sum(c.get("weight", 0) for c in criteria_list)
        }
        
    except Exception as e:
        print(f"Initial JSON parsing failed for prompt {example.get('id', 'Unknown')}: {e}")
        print("Trying to fix with gemini-2.5-flash...")
        
        # Try to fix the JSON with gemini-2.5-flash
        try:
            fixed_response_text = fix_json_with_gemini_flash(response_text)
            rubric_criteria = json.loads(fixed_response_text)
            
            # Convert to the format expected by score_responses.py
            criteria_list = []
            if isinstance(rubric_criteria, list):
                criteria_list = rubric_criteria
            elif isinstance(rubric_criteria, dict):
                # Convert dict format to list format
                for key, value in rubric_criteria.items():
                    if isinstance(value, dict):
                        criterion = value.copy()
                        if "criterion" not in criterion:
                            criterion["criterion"] = key
                        criteria_list.append(criterion)
                    else:
                        criteria_list.append({
                            "criterion": key,
                            "weight": value if isinstance(value, (int, float)) else 1,
                            "description": str(value) if not isinstance(value, (int, float)) else ""
                        })
            
            # Add local_id to each criterion and ensure weights are numeric
            for i, criterion in enumerate(criteria_list):
                if "local_id" not in criterion:
                    criterion["local_id"] = f"c{i+1}"
                # Ensure weight is numeric
                if "weight" not in criterion or not isinstance(criterion.get("weight"), (int, float)):
                    criterion["weight"] = 1
            
            print(f"Successfully fixed JSON for prompt {example.get('id', 'Unknown')}")
            return {
                "id": example["id"],
                "prompt": example["prompt"],
                "original_rubric": rubric_text.strip(),
                "fixed_rubric": fixed_response_text,
                "total_criteria": len(criteria_list),
                "criteria": criteria_list,
                "total_weight": sum(c.get("weight", 0) for c in criteria_list)
            }
            
        except Exception as fix_error:
            print(f"Failed to fix JSON for prompt {example.get('id', 'Unknown')}: {fix_error}")
            return {
                "id": example["id"],
                "prompt": example["prompt"],
                "original_rubric": rubric_text.strip(),
                "error": f"Initial error: {str(e)}; Fix error: {str(fix_error)}",
                "total_criteria": 0,
                "criteria": [],
                "total_weight": 0
            }


def generate_rubrics_without_responses(
    prompts_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    rubric_generator: str = RUBRIC_GENERATOR,
    prompt_template_file: str = PROMPT_TEMPLATE_FILE,
    max_workers: int = NUM_WORKERS,
    backend: str = "api",
    temperature: float = TEMPERATURE,
    max_tokens: Optional[int] = MAX_TOKENS,
    vllm_model: Optional[str] = None,
    vllm_instance: Optional[LLM] = None,
    vllm_tensor_parallel_size: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    vllm_enforce_eager: bool = False,
    max_retries: int = 2,
) -> bool:
    """
    API function to generate rubrics from prompts only (without responses).
    
    Args:
        prompts_file: Path to prompts JSON file
        output_file: Path to save generated rubrics
        sample_size: Number of prompts to sample (None for all)
        rubric_generator: Model to use for rubric generation
        prompt_template_file: Path to rubric generation prompt template
        max_workers: Number of parallel workers
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load data and template
        data = load_json_data(prompts_file)
        prompt_template = load_prompt_template(prompt_template_file)
        
        if sample_size:
            print(f"Sampling {sample_size} prompts...")
            data = random.sample(data, sample_size)
        
        print(f"Processing {len(data)} prompts")
        print("Generating rubrics without responses...")
        rubrics = []

        if backend == "api":
            # Helper to run one API generation pass
            def run_api_pass(examples: List[Dict]) -> List[Dict]:
                results: List[Dict] = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(generate_rubric, ex, prompt_template): ex for ex in examples}
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Generating rubrics"):
                        try:
                            result = future.result()
                        except Exception as worker_error:
                            ex = futures.get(future)
                            ex_id = ex.get("id") if isinstance(ex, dict) else None
                            print(f"Worker failed for prompt {ex_id}: {worker_error}")
                            continue
                        if result is not None:
                            results.append(result)
                return results

            # First pass
            first_pass = run_api_pass(data)
            rubrics.extend(first_pass)

            # Retry failed rubrics up to max_retries times
            last_pass = first_pass
            for retry_round in range(1, max_retries + 1):
                failed_ids = {r.get("id") for r in last_pass if (r.get("error") or not r.get("criteria"))}
                failed_examples = [ex for ex in data if ex.get("id") in failed_ids]
                if not failed_examples:
                    break
                print(f"Retrying failed rubrics (round {retry_round}): {len(failed_examples)} prompts")
                retry_pass = run_api_pass(failed_examples)
                retry_successes = {r.get("id"): r for r in retry_pass if (not r.get("error") and r.get("criteria"))}
                if retry_successes:
                    rubrics = [r for r in rubrics if r.get("id") not in retry_successes]
                    rubrics.extend(retry_successes.values())
                last_pass = retry_pass
        else:
            # vLLM-based batch generation
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Please install vLLM or run with --backend api.")

            model_name = vllm_model or rubric_generator
            if vllm_instance is not None:
                llm = vllm_instance
                should_cleanup = False
            else:
                # Auto-detect GPUs if not provided
                if vllm_tensor_parallel_size is None:
                    try:
                        import torch  # type: ignore
                        vllm_tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
                    except Exception:
                        vllm_tensor_parallel_size = 1

                llm_kwargs = {
                    "model": model_name,
                    "tensor_parallel_size": vllm_tensor_parallel_size,
                    "dtype": "bfloat16",
                    "enforce_eager": vllm_enforce_eager,
                    "trust_remote_code": True,
                }
                if vllm_max_model_len is not None:
                    llm_kwargs["max_model_len"] = vllm_max_model_len
                llm = LLM(**llm_kwargs)
                should_cleanup = True

            try:
                conversations = []
                meta = []
                for ex in data:
                    conv, _ = build_rubric_prompt(ex["prompt"], prompt_template)
                    conversations.append(conv)
                    meta.append(ex)

                sp = SamplingParams(
                    temperature=temperature,
                    max_tokens=(max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else 2048),
                )

                print(f"vLLM generating {len(conversations)} rubrics using model: {model_name}")
                outputs = llm.chat(conversations, sampling_params=sp, use_tqdm=True)

                def process_outputs(examples: List[Dict], outs) -> List[Dict]:
                    processed: List[Dict] = []
                    for i, out in enumerate(outs):
                        example = examples[i]
                        text = out.outputs[0].text if out.outputs else ""
                        try:
                            response_text = text.strip()
                            if response_text.startswith('```json'):
                                response_text = response_text[7:-3].strip()
                            elif response_text.startswith('```'):
                                response_text = response_text[3:-3].strip()
                            rubric_criteria = json.loads(response_text)
                            criteria_list = rubric_criteria if isinstance(rubric_criteria, list) else []
                            for j, criterion in enumerate(criteria_list):
                                if "local_id" not in criterion:
                                    criterion["local_id"] = f"c{j+1}"
                                if "weight" not in criterion or not isinstance(criterion.get("weight"), (int, float)):
                                    criterion["weight"] = 1
                            processed.append({
                                "id": example["id"],
                                "prompt": example["prompt"],
                                "original_rubric": text.strip(),
                                "total_criteria": len(criteria_list),
                                "criteria": criteria_list,
                                "total_weight": sum(c.get("weight", 0) for c in criteria_list)
                            })
                        except Exception as e:
                            print(f"Failed to parse rubric for prompt {example.get('id', 'Unknown')}: {e}")
                            processed.append({
                                "id": example.get("id"),
                                "prompt": example.get("prompt"),
                                "original_rubric": text.strip(),
                                "error": f"Parse error: {e}",
                                "total_criteria": 0,
                                "criteria": [],
                                "total_weight": 0
                            })
                    return processed

                # First pass
                first_pass = process_outputs(meta, outputs)
                rubrics.extend(first_pass)

                # Retry failed up to two additional rounds
                for retry_round in range(1, 3):
                    failed_examples = [ex for ex, r in zip(meta, first_pass) if (r.get("error") or not r.get("criteria"))]
                    if not failed_examples:
                        break
                    conversations = []
                    for ex in failed_examples:
                        conv, _ = build_rubric_prompt(ex["prompt"], prompt_template)
                        conversations.append(conv)
                    print(f"vLLM retry round {retry_round}: {len(failed_examples)} prompts")
                    retry_outs = llm.chat(conversations, sampling_params=sp, use_tqdm=False)
                    retry_results = process_outputs(failed_examples, retry_outs)
                    # Merge successes from retry
                    retry_successes = {r.get("id"): r for r in retry_results if (not r.get("error") and r.get("criteria"))}
                    if retry_successes:
                        rubrics = [r for r in rubrics if r.get("id") not in retry_successes]
                        rubrics.extend(retry_successes.values())
                    # Prepare for potential next round
                    first_pass = retry_results
            finally:
                if vllm_instance is None and 'should_cleanup' in locals() and should_cleanup:
                    try:
                        del llm
                        import torch  # type: ignore
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
        
        print(f"Generated {len(rubrics)} rubrics")
        
        # Calculate summary statistics
        successful_rubrics = [r for r in rubrics if not r.get("error") and r.get("criteria")]
        print(f"Successfully parsed: {len(successful_rubrics)} rubrics")
        
        # Save results (without metadata)
        output_data = {
            "rubrics": successful_rubrics
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(successful_rubrics)} rubrics to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error in generate_rubrics_without_responses: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main workflow execution."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    parser = setup_argparse()
    args = parser.parse_args()

    # Generate default output filename if not specified
    if args.output is None:
        args.output = generate_default_output_filename(
            args.input_file, 
            RUBRIC_GENERATOR, 
            args.sample_size if hasattr(args, 'sample_size') and args.sample_size else None
        )
        print(f"Using default output file: {args.output}")

    # Override generator if --model provided
    rubric_model = args.model if hasattr(args, "model") and args.model else RUBRIC_GENERATOR

    print(f"📊 Using rubric generator: {rubric_model}")
    print(f"📊 Input file: {args.input_file}")
    print(f"📊 Output file: {args.output}")
    print(f"📊 Sample size: {args.sample_size}")
    print(f"📊 Workers: {args.workers}")
    print(f"📊 Backend: {args.backend}")

    # Use the API function to generate rubrics without responses
    success = generate_rubrics_without_responses(
        prompts_file=args.input_file,
        output_file=args.output,
        sample_size=args.sample_size,
        rubric_generator=rubric_model,
        prompt_template_file=PROMPT_TEMPLATE_FILE,
        max_workers=args.workers,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=(None if (not hasattr(args, 'max_tokens') or args.max_tokens == 0) else args.max_tokens),
        vllm_model=rubric_model,
        vllm_instance=None,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_enforce_eager=args.vllm_enforce_eager,
    )
    
    if success:
        print("✅ Rubric generation completed successfully!")
    else:
        print("❌ Rubric generation failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
