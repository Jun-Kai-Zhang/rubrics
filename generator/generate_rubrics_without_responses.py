"""generate_rubrics_without_responses.py

Generate rubrics from prompts only (without responses) using GPT models.

"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# Local imports
from .utils import generate_via_api

# ---------------------------------------------------------------------------
# Configuration defaults -----------------------------------------------------
# ---------------------------------------------------------------------------
RUBRIC_GENERATOR: str = "gemini/gemini-2.5-pro"
MAX_TOKENS: Optional[int] = None
TEMPERATURE: float = 0.0
NUM_WORKERS: int = 64
RANDOM_SEED: int = 42

# Default prompt file path (used in function defaults)
PROMPT_TEMPLATE_FILE = "generator/prompts/generate_rubrics_without_responses_simplified.txt"





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
    temperature: float = TEMPERATURE,
    max_tokens: Optional[int] = MAX_TOKENS,
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

        # Always use API backend
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

        # Identify failures (error present or empty criteria)
        failed_ids = {r.get("id") for r in first_pass if (r.get("error") or not r.get("criteria"))}
        failed_examples = [ex for ex in data if ex.get("id") in failed_ids]
        if failed_examples:
            print(f"Retrying failed rubrics (round 1): {len(failed_examples)} prompts")
            second_pass = run_api_pass(failed_examples)
            # Merge successes from retry by replacing entries for same id where improved
            retry_successes = {r.get("id"): r for r in second_pass if (not r.get("error") and r.get("criteria"))}
            if retry_successes:
                # Drop old entries for these ids
                rubrics = [r for r in rubrics if r.get("id") not in retry_successes]
                rubrics.extend(retry_successes.values())
            # Determine remaining failures for a second retry
            remaining_failed_ids = failed_ids - set(retry_successes.keys())
            remaining_examples = [ex for ex in failed_examples if ex.get("id") in remaining_failed_ids]
            if remaining_examples:
                print(f"Retrying failed rubrics (round 2): {len(remaining_examples)} prompts")
                third_pass = run_api_pass(remaining_examples)
                third_successes = {r.get("id"): r for r in third_pass if (not r.get("error") and r.get("criteria"))}
                if third_successes:
                    rubrics = [r for r in rubrics if r.get("id") not in third_successes]
                    rubrics.extend(third_successes.values())

        
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


