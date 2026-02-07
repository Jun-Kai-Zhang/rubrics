#!/usr/bin/env python3
"""improve_rubrics.py

Improve rubrics using the prompt template and highest scoring responses.
This script finds prompts with tied highest scores and improves rubrics
for ALL such prompts separately, not just globally highest scores.

"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports – prefer package-relative, fall back to path hack for
# "python generator/xyz.py" execution.
# ---------------------------------------------------------------------------
try:
    from .utils import generate_via_api, fix_json_with_gemini_flash, _strip_leading_think_block  # type: ignore
except ImportError:  # pragma: no cover – direct script execution support
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils import generate_via_api, fix_json_with_gemini_flash, _strip_leading_think_block  # type: ignore

# vLLM imports (optional)
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration defaults -----------------------------------------------------
# ---------------------------------------------------------------------------
RUBRIC_IMPROVER_MODEL: str = "gemini/gemini-2.5-pro"
MAX_TOKENS: Optional[int] = None
TEMPERATURE: float = 0.0
RANDOM_SEED: int = 42

# Default file paths
DEFAULT_PROMPT_FILE = "generator/prompts/improve_rubrics.txt"
DEFAULT_SCORED_DATA = "data/exp2/explicit_rubrics_Policy_Model_Qwen2.5_32B_Instruct_Temperature_1.0_TopP_0.95_1000_Prompts_64_Tesponses_Dataset_OST_gemini_gemini-2.5-flash_all_responses_scored.json"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Improve rubrics for all prompts with tied highest scores.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        help="Path to the improvement prompt template (default: %(default)s)",
    )
    parser.add_argument(
        "--scored-data-file",
        type=str,
        default=DEFAULT_SCORED_DATA,
        help="Path to the scored responses JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to the output JSON file (default: auto-generated)",
    )
    parser.add_argument(
        "--previous-rubrics-file",
        type=str,
        default=None,
        help="Path to the previous rubrics file (needed to preserve rubrics for prompts without ties)",
    )
    parser.add_argument(
        "--responses-file", 
        type=str,
        default=None,
        help="Path to the responses file (needed to add prompt text to rubrics)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=RUBRIC_IMPROVER_MODEL,
        help="Model to use for rubric improvement (default: %(default)s)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["api", "vllm"],
        default="api",
        help="Backend to use: 'api' or 'vllm' (local)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Temperature for model generation (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max new tokens to generate (0 to use default)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of concurrent workers for parallel processing (default: %(default)s)",
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
    return parser.parse_args()


def generate_output_filename(input_file: str, model: str) -> str:
    """Generate default output filename based on input file and model."""
    import os
    input_dir = os.path.dirname(input_file)
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    model_clean = model.replace("/", "_").replace("-", "_")
    return os.path.join("data/exp2", f"improved_rubrics_{input_base}_{model_clean}.json")


def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_prompt_template(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()


def find_prompts_with_tied_highest_scores(scored_data: Dict) -> List[Dict]:
    """
    Find all prompts that have tied highest scores within each prompt.
    Returns a list of prompt information with their tied highest scoring responses.
    """
    prompts_with_ties = []
    
    for result in scored_data["results"]:
        prompt_id = result["id"]
        prompt = result["prompt"]
        rubric = result["rubric"]
        responses = result["scored_responses"]
        
        if len(responses) < 2:
            continue
        
        # Get all unique scores to find the absolute highest
        all_scores = [resp["score"] for resp in responses]
        absolute_highest_score = max(all_scores) if all_scores else 0
        
        # Group responses by score within this prompt
        score_groups = defaultdict(list)
        for response in responses:
            score = response["score"]
            score_groups[score].append({
                "prompt": prompt,
                "rubric": rubric,
                "response": response["response"],
                "score_text": response["score_text"],
                "response_idx": response["response_idx"],
                "prompt_id": response["prompt_id"],
                "score": score
            })
        
        # Find highest score within this prompt
        if not score_groups:
            continue
            
        highest_score = max(score_groups.keys())
        highest_responses = score_groups[highest_score]
        
        # Check if there are ties at the highest score
        if len(highest_responses) >= 2:
            # Randomly select 2 responses from the tied highest scoring group
            selected_responses = random.sample(highest_responses, 2)
            
            prompts_with_ties.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "rubric": rubric,
                "highest_score": highest_score,
                "num_tied": len(highest_responses),
                "selected_responses": selected_responses,
                "tie_type": "highest_score_tie",
                "absolute_highest_score": absolute_highest_score
            })
        else:
            print(f"✅ PROMPT {prompt_id}: No ties (1 response with score {highest_score})")
    
    return prompts_with_ties


def _select_top2_per_prompt(scored_data: Dict) -> List[Dict]:
    """Helper: deterministically select top-2 responses per prompt."""
    results: List[Dict] = []
    for result in scored_data.get("results", []):
        prompt_id = result["id"]
        prompt = result["prompt"]
        rubric = result["rubric"]
        responses = result.get("scored_responses", [])
        if len(responses) < 2:
            continue
        top_2 = sorted(responses, key=lambda x: x["score"], reverse=True)[:2]
        selected = [
            {
                "prompt": prompt,
                "rubric": rubric,
                "response": r["response"],
                "score_text": r["score_text"],
                "response_idx": r["response_idx"],
                "prompt_id": r["prompt_id"],
                "score": r["score"],
            }
            for r in top_2
        ]
        results.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "rubric": rubric,
            "highest_score": top_2[0]["score"],
            "num_tied": 2,
            "selected_responses": selected,
            "tie_type": "top2",
            "absolute_highest_score": top_2[0]["score"],
        })
    return results


def build_improvement_prompt(
    prompt: str,
    rubrics: str,
    response1: str,
    response2: str,
    verification1: str,
    verification2: str,
    template: str
) -> str:
    """Build the improvement prompt using the template."""
    return template.format(
        prompt=prompt,
        rubrics=rubrics,
        response1=response1,
        response2=response2
    )


def _extract_first_balanced_json_array(text: str) -> Optional[str]:
    """Extract the first balanced JSON array substring from text.

    Looks for the first '[' and returns the substring up to its matching ']'
    accounting for nested brackets and string literals.
    Returns None if no balanced array is found.
    """
    in_string = False
    escape = False
    depth = 0
    start_idx: Optional[int] = None
    for i, ch in enumerate(text):
        if start_idx is None:
            if ch == '[':
                start_idx = i
                depth = 1
            continue
        # After start
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0 and start_idx is not None:
                return text[start_idx:i+1]
    return None


def _clean_response_and_try_extract_json(raw_text: str) -> str:
    """Best-effort cleanup and JSON extraction from model output.

    - Strips leading <think> blocks
    - Removes surrounding ```json ... ``` or ``` ... ``` fences (anywhere)
    - If still not valid JSON, extracts the first balanced JSON array
    - If still failing, asks a fast model to repair the JSON
    Returns a JSON string that json.loads can parse (or the original text).
    """
    try:
        text = (raw_text or "").strip()
        if not text:
            return text

        # Remove leading <think>...</think> if present
        try:
            text = _strip_leading_think_block(text)
        except Exception:
            pass

        # Remove code fences if present anywhere in the string
        if '```' in text:
            # Prefer the first fenced block; support optional language tag
            start = text.find('```')
            if start != -1:
                # Skip language tag like ```json\n
                after = text[start+3:]
                # If starts with language tag, drop the first line
                nl_idx = after.find('\n')
                if nl_idx != -1 and after[:nl_idx].strip().lower().startswith('json'):
                    after = after[nl_idx+1:]
                end = after.find('```')
                if end != -1:
                    candidate = after[:end].strip()
                    if candidate:
                        text = candidate

        # Fast path: if this loads, return it
        try:
            json.loads(text)
            return text
        except Exception:
            pass

        # Try extracting the first balanced JSON array
        array_str = _extract_first_balanced_json_array(text)
        if array_str is not None:
            try:
                json.loads(array_str)
                return array_str
            except Exception:
                pass

        # As a last resort, try to fix with the fast JSON fixer
        try:
            fixed = fix_json_with_gemini_flash(text)
            if fixed and fixed.strip():
                # Validate fixed JSON
                json.loads(fixed)
                return fixed
        except Exception:
            pass

        return text
    except Exception:
        return raw_text


def improve_rubrics_for_prompt(
    prompt_info: Dict,
    template: str,
    model: str,
    temperature: float,
    existing_criteria_by_id: Dict[str, List[Dict]] = None,
    backend: str = "api",
    vllm_instance: Optional[LLM] = None,
    max_tokens: Optional[int] = None,
) -> Dict:
    """Generate improved rubrics for a single prompt with tied responses.

    Always uses 'improve' mode for full rubric replacement.
    """
    
    prompt_id = prompt_info["prompt_id"]
    prompt = prompt_info["prompt"]
    rubrics = prompt_info["rubric"]
    selected_responses = prompt_info["selected_responses"]
    
    # Extract response data
    response1_data = selected_responses[0]
    response2_data = selected_responses[1]
    
    response1 = response1_data["response"]
    response2 = response2_data["response"]
    verification1 = response1_data["score_text"]
    verification2 = response2_data["score_text"]
    
    # Build the improvement prompt
    improvement_prompt = build_improvement_prompt(
        prompt, rubrics, response1, response2, verification1, verification2, template
    )
    
    messages = [{"role": "user", "content": improvement_prompt}]

    # Generate improved rubrics with a single attempt (no retries)
    max_retries = 1
    retry_delay = 0.0
    improved_rubrics_text = None
    last_error = None

    if backend == "api":
        for attempt in range(max_retries):
            try:
                improved_rubrics_text = generate_via_api(
                    model,
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                break
            except Exception as e:
                last_error = e
                print(f"❌ API call failed for prompt {prompt_id} after {max_retries} attempt(s): {e}")
    else:
        if not VLLM_AVAILABLE:
            return {
                "prompt_id": prompt_id,
                "original_prompt": prompt,
                "original_rubrics": rubrics,
                "response1": response1_data,
                "response2": response2_data,
                "improvement_prompt": improvement_prompt,
                "error": "vLLM not available; install vllm or use --backend api",
                "success": False,
            }
        try:
            llm = vllm_instance
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=(max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else 2048),
            )
            outputs = llm.chat([messages], sampling_params=sp, use_tqdm=False)
            improved_rubrics_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        except Exception as e:
            last_error = e
            print(f"❌ vLLM generation failed for prompt {prompt_id}: {e}")
    
    # If all retries failed, return error
    if improved_rubrics_text is None:
        return {
            "prompt_id": prompt_id,
            "original_prompt": prompt,
            "original_rubrics": rubrics,
            "response1": response1_data,
            "response2": response2_data,
            "improvement_prompt": improvement_prompt,
            "error": f"Failed to generate improved rubrics after {max_retries} attempts: {last_error}",
            "success": False
        }
    
    # Parse the JSON response
    try:
        # Clean and robustly extract JSON content
        response_text = _clean_response_and_try_extract_json(improved_rubrics_text)
        # Parse JSON - always use improve mode (full replacement)
        parsed_criteria = json.loads(response_text)
        new_criteria = parsed_criteria
        all_criteria = parsed_criteria
        
        # Update local_ids for all criteria to ensure they're sequential
        for i, criterion in enumerate(all_criteria):
            criterion["local_id"] = f"c{i+1}"
        
        return {
            "prompt_id": prompt_id,
            "original_prompt": prompt,
            "original_rubrics": rubrics,
            "response1": response1_data,
            "response2": response2_data,
            "improvement_prompt": improvement_prompt,
            "improved_rubrics_raw": improved_rubrics_text,
            "new_criteria": new_criteria,  # Store new criteria separately
            "improved_criteria": all_criteria,  # Combined criteria
            "total_new_criteria": len(new_criteria),
            "total_improved_criteria": len(all_criteria),
            "total_improved_weight": sum(c.get('weight', 0) for c in all_criteria),
            "original_highest_score": prompt_info["highest_score"],
            "num_tied_responses": prompt_info["num_tied"],
            "success": True,
            "response_selection_info": {
                "response1_idx": response1_data["response_idx"],
                "response1_score": response1_data.get("score", prompt_info["highest_score"]),
                "response2_idx": response2_data["response_idx"],
                "response2_score": response2_data.get("score", prompt_info["highest_score"]),
                "response1_text": response1,
                "response2_text": response2,
                "tie_type": prompt_info.get("tie_type", "unknown"),
                "absolute_highest_score": prompt_info.get("absolute_highest_score", prompt_info["highest_score"])
            }
        }
        
    except Exception as e:
        snippet = (improved_rubrics_text or "").strip().replace('\n', ' ')
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"❌ Error parsing improved rubrics for prompt {prompt_id}: {e} | Raw head: {snippet}")
        return {
            "prompt_id": prompt_id,
            "original_prompt": prompt,
            "original_rubrics": rubrics,
            "response1": response1_data,
            "response2": response2_data,
            "improvement_prompt": improvement_prompt,
            "improved_rubrics_raw": improved_rubrics_text,
            "error": str(e),
            "improved_criteria": None,
            "success": False
        }


def save_results(improvement_results: List[Dict], output_file: str, args, prompt_ids_with_ties: List[str]) -> None:
    """Save all improvement results to file in the format expected by score_responses.py."""
    
    # Calculate summary statistics
    successful_improvements = [r for r in improvement_results if r.get("success", False)]
    failed_improvements = [r for r in improvement_results if not r.get("success", False)]
    
    print(f"📊 Processing {len(improvement_results)} improvement results")
    print(f"📊 Successful: {len(successful_improvements)}, Failed: {len(failed_improvements)}")
    
    # Load prompt text from responses file if provided
    prompts_by_id = {}
    if args.responses_file:
        print(f"📋 Loading prompts from responses file: {args.responses_file}")
        with open(args.responses_file, 'r', encoding='utf-8') as f:
            responses_data = json.load(f)
        for prompt_data in responses_data:
            prompts_by_id[prompt_data["id"]] = prompt_data["prompt"]
        print(f"📊 Loaded {len(prompts_by_id)} prompts from responses file")
    
    # Load existing rubrics from previous rubrics file if provided
    existing_rubrics_by_id = {}
    if args.previous_rubrics_file:
        print(f"📋 Loading existing rubrics from: {args.previous_rubrics_file}")
        with open(args.previous_rubrics_file, 'r', encoding='utf-8') as f:
            previous_data = json.load(f)
        for rubric in previous_data.get("rubrics", []):
            existing_rubrics_by_id[rubric["id"]] = rubric
        print(f"📊 Loaded {len(existing_rubrics_by_id)} existing rubrics")
    
    # Create improvements mapping
    improvements_by_prompt = {}
    for result in successful_improvements:
        improvements_by_prompt[result["prompt_id"]] = result
    
    # Build the combined rubrics list
    combined_rubrics = []
    
    for prompt_id in prompt_ids_with_ties:
        if prompt_id in improvements_by_prompt:
            # Use improved rubrics for this prompt
            improvement = improvements_by_prompt[prompt_id]
            improved_criteria = improvement["improved_criteria"]
            original_rubric = improvement.get("original_rubrics", "")
            
            print(f"✅ Using improved rubrics for prompt {prompt_id}")
            
            # Add local_id to each criterion if not present
            for i, criterion in enumerate(improved_criteria):
                if "local_id" not in criterion:
                    criterion["local_id"] = f"c{i+1}"
            
            rubric_entry = {
                "id": prompt_id,
                "original_rubric": original_rubric,
                "total_criteria": len(improved_criteria),
                "criteria": improved_criteria.copy(),
                "total_weight": sum(c.get('weight', 0) for c in improved_criteria)
            }
            
            # Add prompt text if available
            if prompt_id in prompts_by_id:
                rubric_entry["prompt"] = prompts_by_id[prompt_id]
                print(f"📋     Added prompt text for {prompt_id}")
            
            combined_rubrics.append(rubric_entry)
        else:
            # Use existing rubrics for other prompts
            if prompt_id in existing_rubrics_by_id:
                print(f"📋 Using existing rubrics for prompt {prompt_id}")
                existing_rubric = existing_rubrics_by_id[prompt_id].copy()
                
                # Add prompt text if not already present and available
                if "prompt" not in existing_rubric and prompt_id in prompts_by_id:
                    existing_rubric["prompt"] = prompts_by_id[prompt_id]
                    print(f"📋     Added prompt text for existing rubric {prompt_id}")
                
                combined_rubrics.append(existing_rubric)
            else:
                print(f"⚠️  No existing rubrics found for prompt {prompt_id}")
    
    # Create output data in the format expected by score_responses.py
    output_data = {
        "metadata": {
            "total_rubrics_parsed": len(combined_rubrics),
            "source": "improved_rubrics_direct_format",
            "model": args.model,
            "temperature": args.temperature,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "prompt_ids_covered": prompt_ids_with_ties,
            "improved_prompt_ids": list(improvements_by_prompt.keys()),
            "num_prompts_improved": len(improvements_by_prompt),
            "successful_improvements": len(successful_improvements),
            "failed_improvements": len(failed_improvements),
            "previous_rubrics_file": args.previous_rubrics_file,
            "responses_file": args.responses_file,
            "random_seed": RANDOM_SEED,
        },
        "rubrics": combined_rubrics  # This is the format expected by score_responses.py
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved rubrics in score_responses.py format to: {output_file}")
    print(f"📊 Combined rubrics for {len(combined_rubrics)} prompts:")
    for rubric in combined_rubrics:
        prompt_id = rubric["id"]
        criteria_count = rubric.get("total_criteria", 0)
        has_prompt = "prompt" in rubric
        if prompt_id in improvements_by_prompt:
            print(f"📋   {prompt_id}: {criteria_count} criteria (IMPROVED) {'✓ prompt' if has_prompt else '✗ no prompt'}")
        else:
            print(f"📋   {prompt_id}: {criteria_count} criteria (existing) {'✓ prompt' if has_prompt else '✗ no prompt'}")


def improve_rubrics(
    scored_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    rubric_improver_model: str = RUBRIC_IMPROVER_MODEL,
    prompt_template_file: str = DEFAULT_PROMPT_FILE,
    max_workers: int = 16,
    previous_rubrics_file: str = None,
    selection_strategy: str = "ties",
    backend: str = "api",
    vllm_model: Optional[str] = None,
    vllm_tensor_parallel_size: Optional[int] = None,
    vllm_max_model_len: Optional[int] = None,
    vllm_enforce_eager: bool = False,
    temperature: float = TEMPERATURE,
    max_tokens: Optional[int] = MAX_TOKENS,
    vllm_instance: Optional[LLM] = None,
) -> bool:
    """
    API function to improve rubrics using highest scoring responses.

    Always uses 'improve' mode for full rubric replacement.

    Args:
        scored_file: Path to scored responses JSON file
        output_file: Path to save improved rubrics
        sample_size: Number of prompts to sample (None for all)
        rubric_improver_model: Model to use for rubric improvement
        prompt_template_file: Path to improvement prompt template
        max_workers: Number of parallel workers

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Use default improve template if none provided
        if not prompt_template_file:
            prompt_template_file = "generator/prompts/improve_rubrics.txt"
        
        # Load data and template
        scored_data = load_json_data(scored_file)
        prompt_template = load_prompt_template(prompt_template_file)
        
        # Load existing criteria from previous rubrics file if provided
        existing_criteria_by_id = {}
        if previous_rubrics_file:
            print(f"📋 Loading existing criteria from: {previous_rubrics_file}")
            with open(previous_rubrics_file, 'r', encoding='utf-8') as f:
                previous_data = json.load(f)
            for rubric in previous_data.get("rubrics", []):
                existing_criteria_by_id[rubric["id"]] = rubric.get("criteria", [])
            print(f"📊 Loaded existing criteria for {len(existing_criteria_by_id)} prompts")
        
        # Select responses according to selection strategy and force mode
        if selection_strategy == "top2":
            print("Selection strategy: top2 (deterministic top-2 per prompt)")
            prompts_with_ties = _select_top2_per_prompt(scored_data)
        else:
            prompts_with_ties = find_prompts_with_tied_highest_scores(scored_data)
        
        if not prompts_with_ties:
            print("No prompts found with tied highest scores! No improvements needed.")
            # Save empty results
            output_data = {"rubrics": []}
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as fp:
                json.dump(output_data, fp, indent=2, ensure_ascii=False)
            return True
        
        print(f"Found {len(prompts_with_ties)} prompts to improve")
        
        # Prepare vLLM instance if needed
        if backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Please install vLLM or run with backend=api.")
            if vllm_instance is None:
                model_name = vllm_model or rubric_improver_model
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
                }
                if vllm_max_model_len is not None:
                    llm_kwargs["max_model_len"] = vllm_max_model_len
                vllm_instance = LLM(**llm_kwargs)

        improvement_results = []
        if backend == "vllm":
            # Batch all prompts into a single vLLM chat call to avoid concurrent NCCL ops
            conversations: List[List[Dict]] = []
            metas: List[Dict] = []
            for prompt_info in prompts_with_ties:
                prompt_id = prompt_info["prompt_id"]
                prompt = prompt_info["prompt"]
                rubrics = prompt_info["rubric"]
                selected_responses = prompt_info["selected_responses"]
                response1 = selected_responses[0]["response"]
                response2 = selected_responses[1]["response"]
                verification1 = selected_responses[0]["score_text"]
                verification2 = selected_responses[1]["score_text"]
                improvement_prompt = build_improvement_prompt(
                    prompt, rubrics, response1, response2, verification1, verification2, prompt_template
                )
                conversations.append([{"role": "user", "content": improvement_prompt}])
                metas.append(prompt_info)

            sp = SamplingParams(
                temperature=temperature,
                max_tokens=(max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else 2048),
            )
            outputs = vllm_instance.chat(conversations, sampling_params=sp, use_tqdm=True)

            for i, out in enumerate(outputs):
                prompt_info = metas[i]
                prompt_id = prompt_info["prompt_id"]
                prompt = prompt_info["prompt"]
                rubrics_text = prompt_info["rubric"]
                selected_responses = prompt_info["selected_responses"]
                text = out.outputs[0].text if out.outputs else ""
                try:
                    response_text = _clean_response_and_try_extract_json(text)
                    parsed_criteria = json.loads(response_text)
                    # Always use improve mode (full replacement)
                    new_criteria = parsed_criteria
                    all_criteria = parsed_criteria
                    for j, criterion in enumerate(all_criteria):
                        if "local_id" not in criterion:
                            criterion["local_id"] = f"c{j+1}"
                    rubric_entry = {
                        "id": prompt_id,
                        "prompt": prompt,
                        "original_rubric": rubrics_text,
                        "total_criteria": len(all_criteria),
                        "criteria": all_criteria,
                        "total_weight": sum(c.get('weight', 0) for c in all_criteria),
                        "response_selection_info": {
                            "response1_idx": selected_responses[0]["response_idx"],
                            "response1_score": selected_responses[0].get("score", prompt_info["highest_score"]),
                            "response2_idx": selected_responses[1]["response_idx"],
                            "response2_score": selected_responses[1].get("score", prompt_info["highest_score"]),
                            "response1_text": selected_responses[0]["response"],
                            "response2_text": selected_responses[1]["response"],
                            "tie_type": prompt_info.get("tie_type", "unknown"),
                            "absolute_highest_score": prompt_info.get("absolute_highest_score", prompt_info["highest_score"])
                        }
                    }
                    improvement_results.append(rubric_entry)
                except Exception as e:
                    snippet = (text or "").strip().replace('\n', ' ')
                    if len(snippet) > 200:
                        snippet = snippet[:200] + "..."
                    print(f"❌ Error parsing vLLM improved rubrics for prompt {prompt_id}: {e} | Raw head: {snippet}")
        else:
            # API backend: parallelize safely with threads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def submit_batch(items: List[Dict]):
                    return {
                        executor.submit(
                            improve_rubrics_for_prompt,
                            prompt_info,
                            prompt_template,
                            rubric_improver_model,
                            temperature,
                            existing_criteria_by_id,
                            backend,
                            None,
                            (max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else None),
                        ): prompt_info for prompt_info in items
                    }

                # First pass
                futures = submit_batch(prompts_with_ties)
                first_pass_results: Dict[str, Dict] = {}
                failed_prompt_infos: List[Dict] = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Improving rubrics"):
                    result = future.result()
                    pid = result.get("prompt_id") if isinstance(result, dict) else None
                    if result is not None and result.get("success", False):
                        first_pass_results[pid] = result
                    else:
                        failed_prompt_infos.append(futures[future])

                # Add successes from first pass
                for pid, result in first_pass_results.items():
                    improved_criteria = result["improved_criteria"]
                    for i, criterion in enumerate(improved_criteria):
                        if "local_id" not in criterion:
                            criterion["local_id"] = f"c{i+1}"
                    rubric_entry = {
                        "id": result["prompt_id"],
                        "prompt": result["original_prompt"],
                        "original_rubric": result["original_rubrics"],
                        "total_criteria": len(improved_criteria),
                        "criteria": improved_criteria,
                        "total_weight": sum(c.get('weight', 0) for c in improved_criteria),
                        "response_selection_info": result.get("response_selection_info", {})
                    }
                    improvement_results.append(rubric_entry)

                # Up to two additional retry rounds for failures
                for retry_round in range(1, 3):
                    if not failed_prompt_infos:
                        break
                    print(f"Retrying rubric improvements (round {retry_round}): {len(failed_prompt_infos)} prompts")
                    futures = submit_batch(failed_prompt_infos)
                    next_failed: List[Dict] = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Improving rubrics retry {retry_round}"):
                        result = future.result()
                        if result is not None and result.get("success", False):
                            improved_criteria = result["improved_criteria"]
                            for i, criterion in enumerate(improved_criteria):
                                if "local_id" not in criterion:
                                    criterion["local_id"] = f"c{i+1}"
                            rubric_entry = {
                                "id": result["prompt_id"],
                                "prompt": result["original_prompt"],
                                "original_rubric": result["original_rubrics"],
                                "total_criteria": len(improved_criteria),
                                "criteria": improved_criteria,
                                "total_weight": sum(c.get('weight', 0) for c in improved_criteria),
                                "response_selection_info": result.get("response_selection_info", {})
                            }
                            improvement_results.append(rubric_entry)
                        else:
                            next_failed.append(futures[future])
                    failed_prompt_infos = next_failed
        
        # Save results (without metadata)
        output_data = {
            "rubrics": improvement_results
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(output_data, fp, indent=2, ensure_ascii=False)
        
        print(f"Rubric improvement completed. Saved {len(improvement_results)} improved rubrics to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error in improve_rubrics_api: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main workflow execution."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Parse arguments
    args = parse_arguments()
    
    # Generate default output filename if not specified
    if args.output_file is None:
        args.output_file = generate_output_filename(args.scored_data_file, args.model)
    
    print(f"📊 Using improvement model: {args.model}")
    print(f"📊 Temperature: {args.temperature}")
    print(f"📊 Scored data file: {args.scored_data_file}")
    print(f"📊 Output file: {args.output_file}")
    print(f"📊 Max workers: {args.max_workers}")
    
    # Use the API function to improve rubrics
    success = improve_rubrics(
        scored_file=args.scored_data_file,
        output_file=args.output_file,
        sample_size=None,  # No sample size argument in improve_rubrics
        rubric_improver_model=args.model,
        prompt_template_file=args.prompt_file,
        max_workers=args.max_workers,
        previous_rubrics_file=args.previous_rubrics_file,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=(None if args.max_tokens == 0 else args.max_tokens),
    )
    
    if success:
        print("✅ Rubric improvement completed successfully!")
        print(f"Results saved to: {args.output_file}")
    else:
        print("❌ Rubric improvement failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
