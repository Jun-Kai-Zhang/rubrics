#!/usr/bin/env python3
"""score_responses.py
Score responses using rubrics from a separate file and select best responses.
Uses maximum parallelism by scoring all responses from all prompts simultaneously.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
# import pandas as pd  # Not used in this code
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports – prefer package-relative, with fallback for script execution.
# ---------------------------------------------------------------------------
import sys
try:
    from .utils import generate_via_api  # type: ignore
except ImportError:  # pragma: no cover
    print("ImportError: utils.py not found, using path hack")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import generate_via_api  # type: ignore


# Set vLLM environment variable to use v0 API
os.environ["VLLM_USE_V1"] = "0"

# vLLM imports (imported conditionally)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# GPU detection
def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except ImportError:
        # Try alternative methods if PyTorch is not available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n'))
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0


# ---------------------------------------------------------------------------
# Configuration defaults -----------------------------------------------------
# ---------------------------------------------------------------------------
MAX_TOKENS: Optional[int] = None
TEMPERATURE: float = 0.0
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Argument parsing -----------------------------------------------------------
# ---------------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score responses using rubrics from a separate file and select best responses.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["api", "vllm"],
        default="vllm",
        help="Backend to use for scoring: 'api' for API-based scoring, 'vllm' for vLLM batch processing (default: %(default)s)",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model to use for scoring/verification (default: %(default)s)",
    )

    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=None,
        help="Number of GPUs to use for tensor parallelism in vLLM (default: auto-detect all available GPUs)",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Maximum sequence length for vLLM model (default: auto)",
    )
    parser.add_argument(
        "--no-vllm-enforce-eager",
        action="store_true",
        default=False,
        help="Disable vLLM eager mode (enable CUDA graph capture). By default, vLLM runs in eager mode to avoid compatibility issues.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel worker threads to use for API backend (default: %(default)s)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retry attempts for failed parsing (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-temperature",
        type=float,
        default=1.0,
        help="Temperature to use for retry attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of prompts to sample from the dataset (default: process all)",
    )
    parser.add_argument(
        "--responses-per-prompt",
        type=int,
        default=None,
        help="Number of responses per prompt to score (default: score all responses)",
    )
    parser.add_argument(
        "--responses-file",
        type=str,
        default="data/exp0.3/Policy_Model_Qwen2.5_32B_Instruct_Temperature_1.0_TopP_0.95_1000_Prompts_64_Tesponses_Dataset_OST.json",
        help="Path to the file containing responses (default: %(default)s)",
    )
    parser.add_argument(
        "--rubrics-file",
        type=str,
        default="data/exp2/workflow_results_2025-07-02_23-52-58/final_rubrics.json",
        help="Path to the file containing parsed rubrics (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated based on responses file)",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Ignore rubric criterion weights and treat each criterion as weight 1",
    )
    return parser.parse_args()


def generate_output_filename(input_file: str, verifier: str, responses_per_prompt: int = None, backend: str = "api") -> str:
    """Generate default output filename based on input file and verifier."""
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.basename(input_file)
    
    # Remove .json extension and add verifier name and _scored_responses.json
    name_without_ext = os.path.splitext(input_basename)[0]
    
    # Sanitize verifier name for filename (replace / with _)
    verifier_sanitized = verifier.replace("/", "_")
    
    # Add backend information to filename
    backend_suffix = f"_{backend}" if backend != "api" else ""
    
    # Add responses per prompt to filename if specified
    responses_suffix = f"_{responses_per_prompt}responses" if responses_per_prompt else "_all_responses"
    return os.path.join("data/exp2", f"explicit_rubrics_{name_without_ext}_{verifier_sanitized}{backend_suffix}{responses_suffix}_scored_new.json")


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_verifier_prompt(prompt_file: str = "generator/prompts/verifier_explicit.txt") -> str:
    """Load the verifier prompt template from file."""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), prompt_file)
    with open(prompt_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()


def load_rubrics_from_file(rubrics_file_path: str) -> tuple[Dict[str, str], Dict[str, List[Dict]]]:
    """Load rubrics from the parsed rubrics JSON file.
    
    Returns:
        tuple: (rubrics_by_id, criteria_by_id) where rubrics_by_id maps prompt_id to rubric text
               and criteria_by_id maps prompt_id to list of criteria with weights
    """
    print(f"Loading rubrics from {rubrics_file_path}")
    with open(rubrics_file_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    
    # Extract the rubrics array from the data structure
    rubrics_data = data.get("rubrics", [])
    
    # Create mappings from prompt_id to rubric text and criteria
    rubrics_by_id = {}
    criteria_by_id = {}
    
    for entry in rubrics_data:
        prompt_id = entry.get("id")
        raw_criteria_list = entry.get("criteria", [])
        
        # Normalize criteria to ensure required fields exist: criterion (str), weight (int), local_id (str)
        normalized_criteria_list = []
        for idx, item in enumerate(raw_criteria_list):
            if isinstance(item, dict):
                text = item.get("criterion")
                # Fallbacks if 'criterion' is missing or empty
                if not isinstance(text, str) or not text.strip():
                    # Debug print of the problematic dict as requested
                    try:
                        print(
                            f"[RubricNormalization] Missing 'criterion' — prompt_id={prompt_id}, index={idx}, item={json.dumps(item, ensure_ascii=False)}"
                        )
                    except Exception:
                        print(f"[RubricNormalization] Missing 'criterion' — prompt_id={prompt_id}, index={idx}, item={item}")
                    desc = item.get("description")
                    if isinstance(desc, str) and desc.strip():
                        text = desc.strip()
                    else:
                        # Try first string value in dict
                        text = next((v for v in item.values() if isinstance(v, str) and v.strip()), "")
                        if not text:
                            # As a last resort, serialize the dict
                            text = json.dumps(item, ensure_ascii=False)
                weight = item.get("weight", 1)
                try:
                    weight = int(weight)
                except Exception:
                    weight = 1
                local_id = item.get("local_id", f"c{idx+1}")
                normalized_criteria_list.append({
                    "criterion": text,
                    "weight": weight,
                    "local_id": local_id,
                })
            elif isinstance(item, str):
                normalized_criteria_list.append({
                    "criterion": item,
                    "weight": 1,
                    "local_id": f"c{idx+1}",
                })
            else:
                normalized_criteria_list.append({
                    "criterion": str(item),
                    "weight": 1,
                    "local_id": f"c{idx+1}",
                })
        
        # Store normalized list for weight calculation and downstream parsing
        criteria_by_id[prompt_id] = normalized_criteria_list
        
        # Build human-readable rubric text from normalized criteria
        criteria_lines = [f"{c['local_id']}: {c['criterion']}" for c in normalized_criteria_list]
        if not criteria_lines and "original_rubric" in entry:
            rubric_text = entry["original_rubric"]
        else:
            rubric_text = "Evaluate the response based on the following criteria:\n\n" + "\n".join(f"- {c}" for c in criteria_lines)
        
        rubrics_by_id[prompt_id] = rubric_text
    
    print(f"Loaded {len(rubrics_by_id)} rubrics")
    return rubrics_by_id, criteria_by_id


def extract_rubric_from_loaded_data(rubrics_data: Dict[str, str], prompt_id: str) -> str:
    """Extract rubric for a given prompt ID from the loaded rubrics data."""
    if prompt_id in rubrics_data:
        return rubrics_data[prompt_id]
    else:
        # Fallback rubric if not found
        print(f"Warning: No rubric found for prompt ID {prompt_id}, using fallback")
        return "Evaluate the response based on:\n- Relevance to the prompt (weight: 1)\n- Accuracy of information (weight: 1)\n- Clarity and coherence (weight: 1)"


def build_eval_prompt(prompt: str, rubric: str, response: str, verifier_prompt_template: str) -> List[Dict]:
    """Return conversation that asks a model to grade *response* under *rubric*."""
    # Format the verifier prompt template with the actual values
    formatted_prompt = verifier_prompt_template.format(
        prompt=prompt,
        response=response,
        rubric=rubric
    )
    return [
        {"role": "user", "content": formatted_prompt},
    ]


def parse_explicit_verifier_output(verifier_output: str, criteria_by_id: Dict[str, List[Dict]], prompt_id: str, use_weights: bool = True) -> tuple[int, str, Dict, bool]:
    """Parse the explicit verifier output and calculate weighted score.
    
    Returns:
        tuple: (weighted_score, debug_info, parsed_answers, parse_success)
    """
    try:
        # Extract JSON from the output (no longer using \boxed{} format)
        import re
        
        # Strip whitespace and try to find JSON object
        cleaned_output = verifier_output.strip()
        
        # Method 1: Try to extract JSON object directly
        json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Method 2: Look for individual key-value pairs and reconstruct JSON
            json_pattern = re.findall(r'"(c\d+)"\s*:\s*"(yes|no)"', cleaned_output, re.IGNORECASE)
            if json_pattern:
                # Reconstruct JSON from found patterns
                json_str = "{" + ", ".join([f'"{k}":"{v}"' for k, v in json_pattern]) + "}"
            else:
                return 0, f"No parseable JSON found in output: {verifier_output[:200]}...", {}, False
        
        # Parse the JSON
        try:
            answers = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues with a more robust approach
            try:
                # Method 1: Try to extract key-value pairs using regex for c1, c2, etc. format
                pattern = r'"(c\d+)"\s*:\s*"(yes|no)"'
                matches = re.findall(pattern, json_str, re.IGNORECASE)
                if matches:
                    answers = {key: value.lower() for key, value in matches}
                else:
                    # Method 2: Try basic quote cleaning and common JSON fixes
                    cleaned_json = json_str.replace('\\"', '"').replace('""', '"').replace("'", '"')
                    answers = json.loads(cleaned_json)
            except (json.JSONDecodeError, Exception):
                return 0, f"Invalid JSON after all attempts: {json_str[:500]}... | Error: {e}", {}, False
        
        # Get the criteria for this prompt to lookup weights
        if prompt_id not in criteria_by_id:
            return 0, f"No criteria found for prompt {prompt_id}", {}, False
        
        criteria_list = criteria_by_id[prompt_id]
        
        # Create mappings from local_id to weight and criterion text
        weights_by_id = {}
        criteria_text_by_id = {}
        for criterion_data in criteria_list:
            local_id = criterion_data.get("local_id", f"c{criteria_list.index(criterion_data)+1}")
            weight = criterion_data.get("weight", 1)
            criterion_text = criterion_data["criterion"]
            weights_by_id[local_id] = weight
            criteria_text_by_id[local_id] = criterion_text
        
        # Calculate weighted score and build structured parsed answers
        total_score = 0
        total_weight = 0
        debug_parts = []
        missing_criteria = []
        parsed_answers = {}
        
        # First, collect all criterion IDs that should be present
        expected_criteria = set(weights_by_id.keys())
        found_criteria = set(answers.keys())
        
        for criterion_id, answer in answers.items():
            # Get the weight for this criterion
            weight = weights_by_id.get(criterion_id, 1)  # Default to 1 if not found
            effective_weight = weight if use_weights else 1
            criterion_text = criteria_text_by_id.get(criterion_id, "Unknown criterion")
            
            # Store parsed answer with full context
            parsed_answers[criterion_id] = {
                "answer": answer.lower(),
                "weight": weight,
                "effective_weight": effective_weight,
                "criterion_text": criterion_text,
                "found": True
            }
            
            if criterion_id not in weights_by_id:
                debug_parts.append(f"{criterion_id}:{answer}(w={weight},UNKNOWN)")
            else:
                if answer.lower() == "yes":
                    score_contribution = effective_weight
                else:
                    score_contribution = 0
                
                total_score += score_contribution
                total_weight += effective_weight
                debug_parts.append(f"{criterion_id}:{answer}(w={effective_weight})")
        
        # Check for missing criteria
        missing_criteria = expected_criteria - found_criteria
        if missing_criteria:
            # Add missing criteria as "no" (0 score) but include their weights
            for missing_id in missing_criteria:
                weight = weights_by_id[missing_id]
                effective_weight = weight if use_weights else 1
                criterion_text = criteria_text_by_id[missing_id]
                total_weight += effective_weight
                debug_parts.append(f"{missing_id}:MISSING(w={effective_weight})")
                
                # Store missing criteria in parsed answers
                parsed_answers[missing_id] = {
                    "answer": "no",  # Missing criteria count as "no"
                    "weight": weight,
                    "effective_weight": effective_weight,
                    "criterion_text": criterion_text,
                    "found": False
                }
        
        # Use raw weighted score instead of percentage
        raw_score = total_score
        
        # Only output debug info for failed parsing or missing criteria
        debug_info = ""
        if missing_criteria:
            debug_info = f"Missing criteria: {list(missing_criteria)}"
        
        return raw_score, debug_info, parsed_answers, True
        
    except Exception as e:
        return 0, f"Error parsing output: {e} | Raw output: {verifier_output[:500]}...", {}, False


def prepare_all_scoring_tasks(responses_data: List[Dict], rubrics_data: Dict[str, str], criteria_by_id: Dict[str, List[Dict]], verifier_prompt_template: str, verifier: str, responses_per_prompt: int = None, max_retries: int = 2, retry_temperature: float = 1.0, use_weights: bool = True) -> List[tuple]:
    """Prepare all scoring tasks from all prompts for parallel processing."""
    all_tasks = []
    
    for prompt_data in responses_data:
        prompt_id = prompt_data["id"]
        prompt = prompt_data["prompt"]
        rubric = extract_rubric_from_loaded_data(rubrics_data, prompt_id)
        responses = prompt_data["responses"]
        
        # Limit responses to the number specified for scoring
        if responses_per_prompt is not None and responses_per_prompt < len(responses):
            responses = responses[:responses_per_prompt]
        
        # Create tasks for each response in this prompt
        for response_idx, response in enumerate(responses):
            task = (prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier, use_weights)
            all_tasks.append(task)
    
    return all_tasks


def score_single_response_with_prompt_id(args_tuple) -> Dict:
    """Score a single response using the rubric. Includes prompt_id for grouping results."""
    prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier, use_weights = args_tuple
    
    # Track retry attempts
    retry_attempts = []
    
    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        try:
            conv = build_eval_prompt(prompt, rubric, response, verifier_prompt_template)
            
            # Use normal temperature for first attempt, retry temperature for subsequent attempts
            current_temperature = TEMPERATURE if attempt == 0 else retry_temperature
            
            score_text = generate_via_api(
                verifier,
                conv,
                max_tokens=MAX_TOKENS,
                temperature=current_temperature,
            )
            
            # Parse the explicit verifier output and calculate weighted/unweighted score
            score, debug_info, parsed_answers, parse_success = parse_explicit_verifier_output(score_text, criteria_by_id, prompt_id, use_weights=use_weights)
            
            # Record this attempt
            retry_attempts.append({
                "attempt": attempt + 1,
                "temperature": current_temperature,
                "parse_success": parse_success,
                "score": score,
                "debug_info": debug_info
            })
            
            # If parsing was successful, return the result
            if parse_success:
                if attempt > 0:
                    print(f"SUCCESS: Retry attempt {attempt + 1} succeeded for {prompt_id}:{response_idx}")
                
                return {
                    "prompt_id": prompt_id,
                    "response": response,
                    "response_idx": response_idx,
                    "score": score,
                    "score_text": score_text.strip(),
                    "debug_info": debug_info,
                    "parsed_answers": parsed_answers,
                    "_parse_success": parse_success,
                    "retry_attempts": retry_attempts,
                    "final_attempt": attempt + 1,
                }
            else:
                if attempt < max_retries:
                    print(f"RETRY: Attempt {attempt + 1} failed for {prompt_id}:{response_idx}, retrying with temperature {retry_temperature}")
                    if debug_info:
                        print(f"  Debug info: {debug_info}")
                else:
                    print(f"FAILED: All {max_retries + 1} attempts failed for {prompt_id}:{response_idx}")
                    if debug_info:
                        print(f"  Final debug info: {debug_info}")
                        
        except Exception as e:
            print(f"DEBUG: Exception in scoring attempt {attempt + 1} for {prompt_id}:{response_idx}: {e}")
            
            # Record this failed attempt
            retry_attempts.append({
                "attempt": attempt + 1,
                "temperature": TEMPERATURE if attempt == 0 else retry_temperature,
                "parse_success": False,
                "score": 0,
                "debug_info": f"Exception during scoring: {e}",
                "exception": str(e)
            })
            
            if attempt < max_retries:
                print(f"RETRY: Exception on attempt {attempt + 1}, retrying...")
                continue
            else:
                print(f"FAILED: All attempts failed with exceptions for {prompt_id}:{response_idx}")
                import traceback
                traceback.print_exc()
                break
    
    # If we get here, all attempts failed
    return {
        "prompt_id": prompt_id,
        "response": response,
        "response_idx": response_idx,
        "score": 0,
        "score_text": f"All {max_retries + 1} attempts failed",
        "debug_info": f"All {max_retries + 1} attempts failed to parse",
        "parsed_answers": {},
        "_parse_success": False,
        "retry_attempts": retry_attempts,
        "final_attempt": max_retries + 1,
    }


def collect_rubrics_summary(responses_data: List[Dict], rubrics_data: Dict[str, str]) -> Dict:
    """Collect a summary of all rubrics used across prompts."""
    rubrics_by_prompt = {}
    unique_rubrics = set()
    
    for prompt_data in responses_data:
        prompt_id = prompt_data["id"]
        rubric = extract_rubric_from_loaded_data(rubrics_data, prompt_id)
        rubrics_by_prompt[prompt_id] = rubric
        unique_rubrics.add(rubric)
    
    return {
        "total_unique_rubrics": len(unique_rubrics),
        "rubrics_by_prompt_id": rubrics_by_prompt,
        "unique_rubrics": list(unique_rubrics)
    }


# ---------------------------------------------------------------------------
# vLLM batch processing functions -------------------------------------------
# ---------------------------------------------------------------------------
def prepare_vllm_batch_prompts(all_scoring_tasks: List[tuple]) -> List[str]:
    """Prepare all prompts for vLLM batch processing."""
    batch_prompts = []
    for task in all_scoring_tasks:
        prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier, use_weights = task
        conv = build_eval_prompt(prompt, rubric, response, verifier_prompt_template)
        
        # Convert conversation to a single prompt string
        # Assuming conv is a list of dicts with 'role' and 'content' keys
        prompt_text = ""
        for message in conv:
            if message["role"] == "user":
                prompt_text += message["content"]
            elif message["role"] == "assistant":
                prompt_text += message["content"]
        
        batch_prompts.append(prompt_text)
    
    return batch_prompts





def score_responses_with_api(all_scoring_tasks: List[tuple], workers: int) -> List[Dict]:
    """Score all responses using API-based parallel processing."""
    scored_responses = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_single_response_with_prompt_id, task) for task in all_scoring_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring responses"):
            scored_responses.append(future.result())
    return scored_responses


def main() -> None:
    """Main workflow execution."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Parse arguments
    args = parse_arguments()
    
    # Generate default output filename if not specified
    if args.output is None:
        args.output = generate_output_filename(args.responses_file, args.verifier, args.responses_per_prompt, args.backend)
    
    print(f"📊 Using backend: {args.backend}")
    print(f"📊 Verifier model: {args.verifier}")
    print(f"📊 Output file: {args.output}")
    
    # Use the API function to score responses
    success = score_responses_api(
        responses_file=args.responses_file,
        rubrics_file=args.rubrics_file,
        output_file=args.output,
        sample_size=args.sample_size,
        backend=args.backend,
        verifier=args.verifier,
        vllm_instance=None,
        max_retries=args.max_retries,
        retry_temperature=args.retry_temperature,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        max_model_len=args.vllm_max_model_len,
        enforce_eager=not args.no_vllm_enforce_eager,
        use_weights=not args.unweighted,
    )
    
    if success:
        print("✅ Response scoring completed successfully!")
    else:
        print("❌ Response scoring failed!")
        return 1
    
    return 0


# ---------------------------------------------------------------------------
# Direct API function for workflow.py -----------------------------------
# ---------------------------------------------------------------------------

def score_responses_direct(
    responses_file: str,
    rubrics_file: str,
    output_file: str,
    backend: str = "vllm",
    verifier: str = "google/gemma-3-27b-it",
    vllm_instance: Optional[LLM] = None,
    workers: int = 16,
    max_retries: int = 2,
    retry_temperature: float = 1.0,
    sample_size: Optional[int] = None,
    responses_per_prompt: Optional[int] = None,
    filter_prompt_ids: Optional[List[str]] = None,
    vllm_tensor_parallel_size: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    vllm_enforce_eager: bool = True,
    use_weights: bool = True,
) -> bool:
    """
    Direct API function to score responses without subprocess.
    
    Args:
        responses_file: Path to responses JSON file
        rubrics_file: Path to rubrics JSON file
        output_file: Path to output JSON file
        backend: "api" or "vllm"
        verifier: Model name for scoring
        vllm_instance: Pre-initialized vLLM instance (optional, for vllm backend)
        workers: Number of parallel workers for API backend
        max_retries: Maximum retry attempts
        retry_temperature: Temperature for retry attempts
        sample_size: Number of prompts to sample (None for all)
        responses_per_prompt: Number of responses per prompt to score (None for all)
        filter_prompt_ids: List of prompt IDs to filter (None for all)
        vllm_tensor_parallel_size: GPU count for vLLM (only used if vllm_instance is None)
        vllm_max_model_len: Max sequence length for vLLM (only used if vllm_instance is None)
        vllm_enforce_eager: Whether to enforce eager mode (only used if vllm_instance is None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"📊 Scoring responses using {backend} backend")
        print(f"📁 Responses file: {responses_file}")
        print(f"📁 Rubrics file: {rubrics_file}")
        print(f"📁 Output file: {output_file}")
        
        # Load the verifier prompt template
        verifier_prompt_template = load_verifier_prompt()
        
        # Load datasets
        responses_data = load_json_data(responses_file)
        rubrics_data, criteria_by_id = load_rubrics_from_file(rubrics_file)
        
        # Apply sample size if specified
        if sample_size:
            print(f"📊 Sampling {sample_size} prompts...")
            responses_data = random.sample(responses_data, min(sample_size, len(responses_data)))
        
        # Apply prompt ID filtering if specified
        if filter_prompt_ids:
            print(f"📊 Filtering to {len(filter_prompt_ids)} specific prompt IDs...")
            responses_data = [
                prompt_data for prompt_data in responses_data 
                if prompt_data["id"] in filter_prompt_ids
            ]
        
        print(f"📊 Processing {len(responses_data)} prompts")
        
        # Collect rubrics summary
        rubrics_summary = collect_rubrics_summary(responses_data, rubrics_data)
        print(f"📊 Found {rubrics_summary['total_unique_rubrics']} unique rubrics")
        
        # Prepare all scoring tasks
        all_scoring_tasks = prepare_all_scoring_tasks(
            responses_data, 
            rubrics_data, 
            criteria_by_id, 
            verifier_prompt_template, 
            verifier, 
            responses_per_prompt=responses_per_prompt, 
            max_retries=max_retries, 
            retry_temperature=retry_temperature,
            use_weights=use_weights,
        )
        
        total_tasks = len(all_scoring_tasks)
        print(f"📊 Total responses to score: {total_tasks}")
        
        # Score all responses using the selected backend
        if backend == "api":
            scored_responses = score_responses_with_api(all_scoring_tasks, workers)
        elif backend == "vllm":
            scored_responses = score_responses_with_vllm(
                all_scoring_tasks,
                verifier,
                vllm_instance=vllm_instance,
                tensor_parallel_size=vllm_tensor_parallel_size,
                max_model_len=vllm_max_model_len,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                enforce_eager=vllm_enforce_eager,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Calculate retry statistics
        total_final_successes = sum(1 for resp in scored_responses if resp.get("_parse_success", False))
        total_responses = len(scored_responses)
        initial_failures = sum(1 for resp in scored_responses if len(resp.get("retry_attempts", [])) > 1)
        retry_recoveries = sum(1 for resp in scored_responses if len(resp.get("retry_attempts", [])) > 1 and resp.get("_parse_success", False))
        
        print(f"📊 Retry statistics:")
        print(f"  Total responses: {total_responses}")
        print(f"  Initial failures: {initial_failures}")
        print(f"  Retry recoveries: {retry_recoveries}")
        if total_responses > 0:
            print(f"  Final success rate: {(total_final_successes / total_responses * 100):.1f}%")
        else:
            print(f"  Final success rate: N/A (no responses to score)")
        
        # Group results by prompt for better organization
        results_by_prompt = {}
        for scored_response in scored_responses:
            prompt_id = scored_response["prompt_id"]
            if prompt_id not in results_by_prompt:
                results_by_prompt[prompt_id] = []
            results_by_prompt[prompt_id].append(scored_response)
        
        # Create final results with prompt information
        final_results = []
        total_parse_failures = 0
        for prompt_data in responses_data:
            prompt_id = prompt_data["id"]
            prompt = prompt_data["prompt"]
            rubric = extract_rubric_from_loaded_data(rubrics_data, prompt_id)
            
            prompt_scored_responses = results_by_prompt.get(prompt_id, [])
            if prompt_scored_responses:
                # Sort by response index to maintain order
                prompt_scored_responses.sort(key=lambda x: x["response_idx"])
                
                # Count parse failures for this prompt
                prompt_parse_failures = sum(1 for resp in prompt_scored_responses if not resp.get("_parse_success", False))
                total_parse_failures += prompt_parse_failures
                
                # Remove the temporary _parse_success field from responses before saving
                for resp in prompt_scored_responses:
                    resp.pop("_parse_success", None)
                
                # Calculate criterion performance summary
                criterion_summary = {}
                if prompt_scored_responses:
                    sample_parsed_answers = prompt_scored_responses[0].get("parsed_answers", {})
                    for criterion_id, criterion_data in sample_parsed_answers.items():
                        criterion_text = criterion_data.get("criterion_text", "Unknown")
                        weight = criterion_data.get("weight", 1)
                        
                        yes_count = sum(1 for resp in prompt_scored_responses 
                                      if resp.get("parsed_answers", {}).get(criterion_id, {}).get("answer") == "yes")
                        no_count = len(prompt_scored_responses) - yes_count
                        
                        criterion_summary[criterion_id] = {
                            "criterion_text": criterion_text,
                            "weight": weight,
                            "yes_count": yes_count,
                            "no_count": no_count,
                            "yes_percentage": (yes_count / len(prompt_scored_responses)) * 100 if prompt_scored_responses else 0
                        }
                
                result = {
                    "id": prompt_id,
                    "prompt": prompt,
                    "rubric": rubric,
                    "scored_responses": prompt_scored_responses,
                    "num_responses": len(prompt_scored_responses),
                    "score_statistics": {
                        "mean": sum(resp["score"] for resp in prompt_scored_responses) / len(prompt_scored_responses),
                        "min": min(resp["score"] for resp in prompt_scored_responses),
                        "max": max(resp["score"] for resp in prompt_scored_responses),
                    },
                    "criterion_performance_summary": criterion_summary
                }
                final_results.append(result)
        
        # Save results
        output_data = {
            "metadata": {
                "verifier": verifier,
                "scoring_method": "explicit_weighted_scoring",
                "scoring_description": "Score is calculated by parsing yes/no answers for each criterion and applying weights from the rubric",
                "responses_per_prompt": responses_per_prompt,
                "total_prompts": len(final_results),
                "total_responses_scored": len(scored_responses),
                "random_seed": RANDOM_SEED,
                "responses_file": responses_file,
                "rubrics_file": rubrics_file,
                "rubrics_summary": rubrics_summary,
                "retry_configuration": {
                    "max_retries": max_retries,
                    "retry_temperature": retry_temperature,
                    "initial_temperature": TEMPERATURE,
                },
                "retry_statistics": {
                    "total_responses": total_responses,
                    "initial_failures": initial_failures,
                    "retry_recoveries": retry_recoveries,
                    "final_success_rate": (total_final_successes / total_responses * 100) if total_responses > 0 else 0,
                    "retry_recovery_rate": (retry_recoveries / initial_failures * 100) if initial_failures > 0 else 0,
                },
                "use_weights": use_weights,
            },
            "results": final_results
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2)
        
        print(f"✅ Saved results for {len(final_results)} prompts to {output_file}")
        print(f"📊 Total responses scored: {len(scored_responses)}")
        print(f"📊 Total parse failures: {total_parse_failures}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in score_responses_direct: {e}")
        import traceback
        traceback.print_exc()
        return False


def score_responses_with_vllm(
    all_scoring_tasks: List[tuple],
    verifier: str,
    vllm_instance: Optional[LLM] = None,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 1.0,
    enforce_eager: bool = False,
) -> List[Dict]:
    """
    Score all responses using vLLM batch processing with optional pre-initialized instance.
    
    Args:
        all_scoring_tasks: List of scoring tasks
        verifier: Model name
        vllm_instance: Pre-initialized vLLM instance (optional)
        tensor_parallel_size: GPU count (only used if vllm_instance is None)
        max_model_len: Max sequence length (only used if vllm_instance is None)
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        enforce_eager: Whether to enforce eager mode (only used if vllm_instance is None)
    
    Returns:
        List of scored responses
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Please install it with: pip install vllm")
    
    # Use provided vLLM instance or create new one
    if vllm_instance is not None:
        print(f"📊 Using pre-initialized vLLM instance")
        llm = vllm_instance
        should_cleanup = False
    else:
        print(f"📊 Initializing new vLLM instance with model: {verifier}")
        llm_kwargs = {
            "model": verifier,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
            "dtype": "bfloat16",
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        llm = LLM(**llm_kwargs)
        should_cleanup = True
    
    # Prepare sampling parameters for initial attempt
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens or 2048,
        stop=None,
    )
    
    # Prepare batch prompts for initial attempt
    print("📊 Preparing batch prompts for vLLM...")
    batch_prompts = prepare_vllm_batch_prompts(all_scoring_tasks)
    
    print(f"📊 Running vLLM inference on {len(batch_prompts)} prompts...")
    
    # Generate responses using vLLM batch processing
    outputs = llm.generate(batch_prompts, sampling_params)
    
    # Process results and collect failed attempts for retry
    scored_responses = []
    failed_tasks = []
    
    for i, output in enumerate(tqdm(outputs, desc="Processing vLLM outputs")):
        task = all_scoring_tasks[i]
        prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier_model, use_weights = task
        
        try:
            # Get the generated text
            generated_text = output.outputs[0].text
            
            # Parse the explicit verifier output and calculate weighted/unweighted score
            score, debug_info, parsed_answers, parse_success = parse_explicit_verifier_output(
                generated_text, criteria_by_id, prompt_id, use_weights=use_weights
            )
            
            # Track retry attempts
            retry_attempts = [{
                "attempt": 1,
                "temperature": temperature,
                "parse_success": parse_success,
                "score": score,
                "debug_info": debug_info
            }]
            
            if parse_success:
                # Success on first attempt
                scored_responses.append({
                    "prompt_id": prompt_id,
                    "response": response,
                    "response_idx": response_idx,
                    "score": score,
                    "score_text": generated_text.strip(),
                    "debug_info": debug_info,
                    "parsed_answers": parsed_answers,
                    "_parse_success": parse_success,
                    "retry_attempts": retry_attempts,
                    "final_attempt": 1,
                })
            else:
                # Failed on first attempt, add to retry list
                if max_retries > 0:
                    failed_tasks.append({
                        "task": task,
                        "task_index": i,
                        "retry_attempts": retry_attempts,
                        "last_score_text": generated_text.strip()
                    })
                    print(f"RETRY: Initial attempt failed for {prompt_id}:{response_idx}, will retry with temperature {retry_temperature}")
                    if debug_info:
                        print(f"  Debug info: {debug_info}")
                else:
                    # No retries allowed
                    scored_responses.append({
                        "prompt_id": prompt_id,
                        "response": response,
                        "response_idx": response_idx,
                        "score": score,
                        "score_text": generated_text.strip(),
                        "debug_info": debug_info,
                        "parsed_answers": parsed_answers,
                        "_parse_success": parse_success,
                        "retry_attempts": retry_attempts,
                        "final_attempt": 1,
                    })
                    print(f"FAILED: No retries allowed for {prompt_id}:{response_idx}")
            
        except Exception as e:
            print(f"DEBUG: Exception in processing vLLM output for {prompt_id}:{response_idx}: {e}")
            
            # Track retry attempts for exception case
            retry_attempts = [{
                "attempt": 1,
                "temperature": temperature,
                "parse_success": False,
                "score": 0,
                "debug_info": f"Exception during vLLM processing: {e}",
                "exception": str(e)
            }]
            
            if max_retries > 0:
                failed_tasks.append({
                    "task": task,
                    "task_index": i,
                    "retry_attempts": retry_attempts,
                    "last_score_text": f"Error: {e}"
                })
                print(f"RETRY: Exception on initial attempt for {prompt_id}:{response_idx}, will retry")
            else:
                scored_responses.append({
                    "prompt_id": prompt_id,
                    "response": response,
                    "response_idx": response_idx,
                    "score": 0,
                    "score_text": f"Error: {e}",
                    "debug_info": f"Exception during vLLM processing: {e}",
                    "parsed_answers": {},
                    "_parse_success": False,
                    "retry_attempts": retry_attempts,
                    "final_attempt": 1,
                })
    
    # Handle retries for failed tasks
    if failed_tasks:
        print(f"📊 Processing {len(failed_tasks)} failed tasks for retry...")
        
        # Prepare retry sampling parameters with higher temperature
        retry_sampling_params = SamplingParams(
            temperature=retry_temperature,
            max_tokens=max_tokens or 2048,
            stop=None,
        )
        
        # Process retries
        for retry_attempt in range(2, max_retries + 2):  # Start from attempt 2
            if not failed_tasks:
                break
                
            print(f"📊 Retry attempt {retry_attempt} for {len(failed_tasks)} failed tasks...")
            
            # Prepare batch prompts for retry
            retry_batch_prompts = []
            for failed_task in failed_tasks:
                task = failed_task["task"]
                prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier_model, use_weights = task
                conv = build_eval_prompt(prompt, rubric, response, verifier_prompt_template)
                
                # Convert conversation to a single prompt string
                prompt_text = ""
                for message in conv:
                    if message["role"] == "user":
                        prompt_text += message["content"]
                    elif message["role"] == "assistant":
                        prompt_text += message["content"]
                
                retry_batch_prompts.append(prompt_text)
            
            # Generate retry responses
            retry_outputs = llm.generate(retry_batch_prompts, retry_sampling_params)
            
            # Process retry results
            still_failed_tasks = []
            for j, retry_output in enumerate(retry_outputs):
                failed_task = failed_tasks[j]
                task = failed_task["task"]
                prompt, rubric, response, response_idx, verifier_prompt_template, prompt_id, criteria_by_id, max_retries, retry_temperature, verifier_model, use_weights = task
                
                try:
                    # Get the generated text
                    generated_text = retry_output.outputs[0].text
                    
                    # Parse the explicit verifier output and calculate weighted/unweighted score
                    score, debug_info, parsed_answers, parse_success = parse_explicit_verifier_output(
                        generated_text, criteria_by_id, prompt_id, use_weights=use_weights
                    )
                    
                    # Add this retry attempt to the history
                    failed_task["retry_attempts"].append({
                        "attempt": retry_attempt,
                        "temperature": retry_temperature,
                        "parse_success": parse_success,
                        "score": score,
                        "debug_info": debug_info
                    })
                    
                    if parse_success:
                        # Success on retry attempt
                        print(f"SUCCESS: Retry attempt {retry_attempt} succeeded for {prompt_id}:{response_idx}")
                        scored_responses.append({
                            "prompt_id": prompt_id,
                            "response": response,
                            "response_idx": response_idx,
                            "score": score,
                            "score_text": generated_text.strip(),
                            "debug_info": debug_info,
                            "parsed_answers": parsed_answers,
                            "_parse_success": parse_success,
                            "retry_attempts": failed_task["retry_attempts"],
                            "final_attempt": retry_attempt,
                        })
                    else:
                        # Still failed, check if we have more retries
                        if retry_attempt < max_retries + 1:
                            still_failed_tasks.append({
                                "task": task,
                                "task_index": failed_task["task_index"],
                                "retry_attempts": failed_task["retry_attempts"],
                                "last_score_text": generated_text.strip()
                            })
                            print(f"RETRY: Attempt {retry_attempt} failed for {prompt_id}:{response_idx}, will retry again")
                            if debug_info:
                                print(f"  Debug info: {debug_info}")
                        else:
                            # No more retries, mark as final failure
                            print(f"FAILED: All {max_retries + 1} attempts failed for {prompt_id}:{response_idx}")
                            scored_responses.append({
                                "prompt_id": prompt_id,
                                "response": response,
                                "response_idx": response_idx,
                                "score": 0,
                                "score_text": f"All {max_retries + 1} attempts failed",
                                "debug_info": f"All {max_retries + 1} attempts failed to parse",
                                "parsed_answers": {},
                                "_parse_success": False,
                                "retry_attempts": failed_task["retry_attempts"],
                                "final_attempt": retry_attempt,
                            })
                
                except Exception as e:
                    print(f"DEBUG: Exception in retry attempt {retry_attempt} for {prompt_id}:{response_idx}: {e}")
                    
                    # Add this failed retry attempt to the history
                    failed_task["retry_attempts"].append({
                        "attempt": retry_attempt,
                        "temperature": retry_temperature,
                        "parse_success": False,
                        "score": 0,
                        "debug_info": f"Exception during retry: {e}",
                        "exception": str(e)
                    })
                    
                    if retry_attempt < max_retries + 1:
                        still_failed_tasks.append({
                            "task": task,
                            "task_index": failed_task["task_index"],
                            "retry_attempts": failed_task["retry_attempts"],
                            "last_score_text": f"Error: {e}"
                        })
                        print(f"RETRY: Exception on retry attempt {retry_attempt} for {prompt_id}:{response_idx}, will retry again")
                    else:
                        # No more retries, mark as final failure
                        print(f"FAILED: All attempts failed with exceptions for {prompt_id}:{response_idx}")
                        scored_responses.append({
                            "prompt_id": prompt_id,
                            "response": response,
                            "response_idx": response_idx,
                            "score": 0,
                            "score_text": f"All {max_retries + 1} attempts failed",
                            "debug_info": f"All {max_retries + 1} attempts failed with exceptions",
                            "parsed_answers": {},
                            "_parse_success": False,
                            "retry_attempts": failed_task["retry_attempts"],
                            "final_attempt": retry_attempt,
                        })
            
            # Update failed tasks list for next iteration
            failed_tasks = still_failed_tasks
    
    # Clean up vLLM resources only if we created the instance
    if should_cleanup:
        del llm
    
    return scored_responses


def score_responses(
    responses_file: str,
    rubrics_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    backend: str = "vllm",
    verifier: str = "google/gemma-3-27b-it",
    vllm_instance: Optional[LLM] = None,
    max_retries: int = 2,
    retry_temperature: float = 1.0,
    tensor_parallel_size: Optional[int] = None,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = True,
    use_weights: bool = True,
) -> bool:
    """
    API function to score responses using rubrics.
    
    Args:
        responses_file: Path to responses JSON file
        rubrics_file: Path to rubrics JSON file
        output_file: Path to output JSON file
        sample_size: Number of prompts to sample (None for all)
        backend: "api" or "vllm"
        verifier: Model name for scoring
        vllm_instance: Pre-initialized vLLM instance (optional)
        max_retries: Maximum retry attempts
        retry_temperature: Temperature for retry attempts
        tensor_parallel_size: GPU count for vLLM
        max_model_len: Max sequence length for vLLM
        enforce_eager: Whether to enforce eager mode
    
    Returns:
        bool: True if successful, False otherwise
    """
    return score_responses_direct(
        responses_file=responses_file,
        rubrics_file=rubrics_file,
        output_file=output_file,
        backend=backend,
        verifier=verifier,
        vllm_instance=vllm_instance,
        workers=128,
        max_retries=max_retries,
        retry_temperature=retry_temperature,
        sample_size=sample_size,
        responses_per_prompt=None,
        filter_prompt_ids=None,
        vllm_tensor_parallel_size=tensor_parallel_size,
        vllm_max_model_len=max_model_len,
        vllm_enforce_eager=enforce_eager,
        use_weights=use_weights,
    )


if __name__ == "__main__":
    main() 