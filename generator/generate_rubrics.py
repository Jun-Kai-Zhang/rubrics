"""generate_rubrics.py

Generate rubrics from prompts only (without responses) using an
OpenAI-compatible API (litellm proxy, vLLM server, etc.).
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .utils import generate_via_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt_template(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rubric_prompt(prompt: str, template: str) -> tuple[List[Dict], str]:
    """Return (conversation, user_msg) for rubric generation."""
    user_msg = template.format(prompt=prompt)
    return [{"role": "user", "content": user_msg}], user_msg


def fix_json_with_gemini_flash(
    malformed_json: str,
    base_url: Optional[str] = None,
) -> str:
    """Try to fix malformed JSON using gemini-2.5-flash."""
    fix_prompt = (
        "The following text should be a valid JSON but has formatting issues. "
        "Please fix it and return only the valid JSON:\n\n"
        f"{malformed_json}\n\n"
        "Return only the corrected JSON, no explanations or markdown formatting."
    )
    conv = [{"role": "user", "content": fix_prompt}]

    fixed_response = generate_via_api(
        "gemini/gemini-2.5-flash",
        conv,
        max_tokens=None,
        temperature=0.0,
        base_url=base_url,
    )

    response_text = fixed_response.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith("```"):
        response_text = response_text[3:-3].strip()

    return response_text


def _parse_rubric_criteria(response_text: str) -> List[Dict]:
    """Parse raw model text into a list of rubric criteria dicts."""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    rubric_criteria = json.loads(text)

    if isinstance(rubric_criteria, list):
        criteria_list = rubric_criteria
    else:
        raise ValueError(f"Invalid rubric format: {type(rubric_criteria)}")

    for i, criterion in enumerate(criteria_list):
        if "local_id" not in criterion:
            criterion["local_id"] = f"c{i + 1}"
        if "weight" not in criterion or not isinstance(
            criterion.get("weight"), (int, float)
        ):
            criterion["weight"] = 1

    return criteria_list


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_rubric(
    example: Dict,
    template: str,
    model: str = "gemini/gemini-2.5-pro",
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
) -> Dict:
    """Generate a rubric for a single example via the API."""
    prompt = example["prompt"]
    conv, _ = build_rubric_prompt(prompt, template)

    rubric_text = generate_via_api(
        model,
        conv,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
    )

    # First parsing attempt
    try:
        criteria_list = _parse_rubric_criteria(rubric_text)
        return {
            "id": example["id"],
            "prompt": example["prompt"],
            "original_rubric": rubric_text.strip(),
            "total_criteria": len(criteria_list),
            "criteria": criteria_list,
            "total_weight": sum(c.get("weight", 0) for c in criteria_list),
        }
    except Exception as e:
        print(
            f"Initial JSON parsing failed for prompt "
            f"{example.get('id', 'Unknown')}: {e}"
        )
        print("Trying to fix with gemini-2.5-flash...")

    # Fallback: ask gemini-2.5-flash to fix the JSON
    fixed_text = fix_json_with_gemini_flash(rubric_text.strip(), base_url=base_url)
    rubric_criteria = json.loads(fixed_text)

    criteria_list = []
    if isinstance(rubric_criteria, list):
        criteria_list = rubric_criteria
    elif isinstance(rubric_criteria, dict):
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
                    "description": (
                        str(value) if not isinstance(value, (int, float)) else ""
                    ),
                })

    for i, criterion in enumerate(criteria_list):
        if "local_id" not in criterion:
            criterion["local_id"] = f"c{i + 1}"
        if "weight" not in criterion or not isinstance(
            criterion.get("weight"), (int, float)
        ):
            criterion["weight"] = 1

    print(f"Successfully fixed JSON for prompt {example.get('id', 'Unknown')}")
    return {
        "id": example["id"],
        "prompt": example["prompt"],
        "original_rubric": rubric_text.strip(),
        "fixed_rubric": fixed_text,
        "total_criteria": len(criteria_list),
        "criteria": criteria_list,
        "total_weight": sum(c.get("weight", 0) for c in criteria_list),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_rubrics(
    prompts_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    rubric_generator: str = "gemini/gemini-2.5-pro",
    prompt_template_file: str = "generator/prompts/generate_rubrics.txt",
    max_workers: int = 64,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    max_retries: int = 2,
    base_url: Optional[str] = None,
) -> bool:
    """Generate rubrics from prompts only (without responses).

    All model calls go through ``generate_via_api`` → the configured
    OpenAI-compatible endpoint (litellm proxy, vLLM server, etc.).

    Returns True on success.
    """
    data = load_json_data(prompts_file)
    prompt_template = load_prompt_template(prompt_template_file)

    if sample_size:
        print(f"Sampling {sample_size} prompts...")
        data = random.sample(data, sample_size)

    print(f"Processing {len(data)} prompts")
    print("Generating rubrics without responses...")

    # -- helper: one parallel pass over a list of examples ----------------
    def run_pass(examples: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    generate_rubric,
                    ex,
                    prompt_template,
                    model=rubric_generator,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=base_url,
                ): ex
                for ex in examples
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating rubrics",
            ):
                ex = futures[future]
                try:
                    result = future.result()
                except Exception as worker_error:
                    print(
                        f"Worker failed for prompt "
                        f"{ex.get('id', 'Unknown')}: {worker_error}"
                    )
                    result = {
                        "id": ex.get("id"),
                        "prompt": ex.get("prompt"),
                        "original_rubric": "",
                        "error": f"Worker error: {worker_error}",
                        "total_criteria": 0,
                        "criteria": [],
                        "total_weight": 0,
                    }
                results.append(result)
        return results

    # -- first pass -------------------------------------------------------
    rubrics = run_pass(data)

    # -- retry failed rubrics ---------------------------------------------
    last_pass = rubrics
    for retry_round in range(1, max_retries + 1):
        failed_ids = {
            r.get("id")
            for r in last_pass
            if r.get("error") or not r.get("criteria")
        }
        failed_examples = [ex for ex in data if ex.get("id") in failed_ids]
        if not failed_examples:
            break
        print(
            f"Retrying failed rubrics (round {retry_round}): "
            f"{len(failed_examples)} prompts"
        )
        retry_pass = run_pass(failed_examples)
        retry_successes = {
            r.get("id"): r
            for r in retry_pass
            if not r.get("error") and r.get("criteria")
        }
        if retry_successes:
            rubrics = [r for r in rubrics if r.get("id") not in retry_successes]
            rubrics.extend(retry_successes.values())
        last_pass = retry_pass

    # -- save results -----------------------------------------------------
    successful_rubrics = [
        r for r in rubrics if not r.get("error") and r.get("criteria")
    ]
    print(f"Generated {len(rubrics)} rubrics")
    print(f"Successfully parsed: {len(successful_rubrics)} rubrics")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(successful_rubrics, fp, indent=2, ensure_ascii=False)

    print(f"Saved {len(successful_rubrics)} rubrics to {output_file}")
    return True
