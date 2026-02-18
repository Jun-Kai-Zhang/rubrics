"""improve_rubrics.py

Improve rubrics using the prompt template and top-2 scoring responses.
For each prompt, deterministically selects the two highest-scoring
responses and uses them to generate an improved rubric.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .utils import generate_via_api, fix_json_with_gemini_flash, _strip_leading_think_block


def load_json_data(file_path: str) -> Dict:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def load_prompt_template(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()


def select_top2_per_prompt(scored_data: List[Dict]) -> List[Dict]:
    """Deterministically select top-2 responses per prompt.

    For each prompt with at least 2 scored responses, selects the two
    highest-scoring ones to use as reference for rubric improvement.

    Also works with unscored data (single mode) where responses have no
    score_text or response_idx — those fields default gracefully.
    """
    results: List[Dict] = []
    for result in scored_data:
        prompt_id = result["id"]
        prompt = result["prompt"]
        rubric = result.get("rubric", "")
        responses = result.get("scored_responses", [])
        if len(responses) < 2:
            continue
        top_2 = sorted(responses, key=lambda x: x.get("score", 0), reverse=True)[:2]
        selected = [
            {
                "prompt": prompt,
                "rubric": rubric,
                "response": r["response"],
                "score_text": r.get("score_text", ""),
                "response_idx": r.get("response_idx", i),
                "prompt_id": r.get("prompt_id", prompt_id),
                "score": r.get("score", 0),
            }
            for i, r in enumerate(top_2)
        ]
        results.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "rubric": rubric,
            "highest_score": top_2[0].get("score", 0),
            "selected_responses": selected,
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
    """Extract the first balanced JSON array substring from text."""
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
    """Best-effort cleanup and JSON extraction from model output."""
    try:
        text = (raw_text or "").strip()
        if not text:
            return text

        try:
            text = _strip_leading_think_block(text)
        except Exception:
            pass

        # Remove code fences if present
        if '```' in text:
            start = text.find('```')
            if start != -1:
                after = text[start+3:]
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
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
) -> Dict:
    """Generate improved rubrics for a single prompt using its top-2 responses."""
    prompt_id = prompt_info["prompt_id"]
    prompt = prompt_info["prompt"]
    rubrics = prompt_info["rubric"]
    selected_responses = prompt_info["selected_responses"]

    response1_data = selected_responses[0]
    response2_data = selected_responses[1]

    response1 = response1_data["response"]
    response2 = response2_data["response"]
    verification1 = response1_data.get("score_text", "")
    verification2 = response2_data.get("score_text", "")

    improvement_prompt = build_improvement_prompt(
        prompt, rubrics, response1, response2, verification1, verification2, template
    )

    messages = [{"role": "user", "content": improvement_prompt}]

    improved_rubrics_text = None
    last_error = None

    try:
        improved_rubrics_text = generate_via_api(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=base_url,
        )
    except Exception as e:
        last_error = e
        print(f"API call failed for prompt {prompt_id}: {e}")

    if improved_rubrics_text is None:
        return {
            "prompt_id": prompt_id,
            "original_prompt": prompt,
            "original_rubrics": rubrics,
            "response1": response1_data,
            "response2": response2_data,
            "improvement_prompt": improvement_prompt,
            "error": f"Failed to generate improved rubrics: {last_error}",
            "success": False
        }

    # Parse the JSON response
    try:
        response_text = _clean_response_and_try_extract_json(improved_rubrics_text)
        parsed_criteria = json.loads(response_text)
        all_criteria = parsed_criteria

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
            "new_criteria": parsed_criteria,
            "improved_criteria": all_criteria,
            "total_new_criteria": len(parsed_criteria),
            "total_improved_criteria": len(all_criteria),
            "total_improved_weight": sum(c.get('weight', 0) for c in all_criteria),
            "original_highest_score": prompt_info["highest_score"],
            "success": True,
            "response_selection_info": {
                "response1_idx": response1_data["response_idx"],
                "response1_score": response1_data.get("score", prompt_info["highest_score"]),
                "response2_idx": response2_data["response_idx"],
                "response2_score": response2_data.get("score", prompt_info["highest_score"]),
                "response1_text": response1,
                "response2_text": response2,
            }
        }

    except Exception as e:
        snippet = (improved_rubrics_text or "").strip().replace('\n', ' ')
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"Error parsing improved rubrics for prompt {prompt_id}: {e} | Raw head: {snippet}")
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


def improve_rubrics(
    scored_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    rubric_improver_model: str = "gemini/gemini-2.5-pro",
    prompt_template_file: str = "generator/prompts/improve_rubrics.txt",
    max_workers: int = 16,
    previous_rubrics_file: str = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
) -> bool:
    """Improve rubrics using the top-2 scoring responses per prompt.

    For each prompt, selects the two highest-scoring responses and uses
    them to generate an improved rubric via the model.

    Returns True on success.
    """
    if not prompt_template_file:
        prompt_template_file = "generator/prompts/improve_rubrics.txt"

    scored_data = load_json_data(scored_file)
    prompt_template = load_prompt_template(prompt_template_file)

    existing_criteria_by_id = {}
    if previous_rubrics_file:
        print(f"Loading existing criteria from: {previous_rubrics_file}")
        with open(previous_rubrics_file, 'r', encoding='utf-8') as f:
            previous_data = json.load(f)
        for rubric in previous_data:
            existing_criteria_by_id[rubric["id"]] = rubric.get("criteria", [])
        print(f"Loaded existing criteria for {len(existing_criteria_by_id)} prompts")

    prompts_to_improve = select_top2_per_prompt(scored_data)

    if not prompts_to_improve:
        print("No prompts with at least 2 responses found. No improvements needed.")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump([], fp, indent=2, ensure_ascii=False)
        return True

    print(f"Found {len(prompts_to_improve)} prompts to improve")

    improvement_results = []
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
                    (max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else None),
                    base_url,
                ): prompt_info for prompt_info in items
            }

        # First pass
        futures = submit_batch(prompts_to_improve)
        first_pass_results: Dict[str, Dict] = {}
        failed_prompt_infos: List[Dict] = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Improving rubrics"):
            result = future.result()
            pid = result.get("prompt_id") if isinstance(result, dict) else None
            if result is not None and result.get("success", False):
                first_pass_results[pid] = result
            else:
                failed_prompt_infos.append(futures[future])

        for pid, result in first_pass_results.items():
            improved_criteria = result["improved_criteria"]
            for i, criterion in enumerate(improved_criteria):
                if "local_id" not in criterion:
                    criterion["local_id"] = f"c{i+1}"
            improvement_results.append({
                "id": result["prompt_id"],
                "prompt": result["original_prompt"],
                "original_rubric": result["original_rubrics"],
                "total_criteria": len(improved_criteria),
                "criteria": improved_criteria,
                "total_weight": sum(c.get('weight', 0) for c in improved_criteria),
                "response_selection_info": result.get("response_selection_info", {})
            })

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
                    improvement_results.append({
                        "id": result["prompt_id"],
                        "prompt": result["original_prompt"],
                        "original_rubric": result["original_rubrics"],
                        "total_criteria": len(improved_criteria),
                        "criteria": improved_criteria,
                        "total_weight": sum(c.get('weight', 0) for c in improved_criteria),
                        "response_selection_info": result.get("response_selection_info", {})
                    })
                else:
                    next_failed.append(futures[future])
            failed_prompt_infos = next_failed

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(improvement_results, fp, indent=2, ensure_ascii=False)

    print(f"Rubric improvement completed. Saved {len(improvement_results)} improved rubrics to {output_file}")
    return True
