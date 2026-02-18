"""score_responses.py

Score responses using rubrics from a separate file and select best responses.
Uses maximum parallelism by scoring all responses from all prompts simultaneously.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .utils import generate_via_api


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
        rubrics_data = json.load(fp)

    rubrics_by_id = {}
    criteria_by_id = {}

    for entry in rubrics_data:
        prompt_id = entry.get("id")
        raw_criteria_list = entry.get("criteria", [])

        # Normalize criteria to ensure required fields exist
        normalized_criteria_list = []
        for idx, item in enumerate(raw_criteria_list):
            if isinstance(item, dict):
                text = item.get("criterion")
                if not isinstance(text, str) or not text.strip():
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
                        text = next((v for v in item.values() if isinstance(v, str) and v.strip()), "")
                        if not text:
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

        criteria_by_id[prompt_id] = normalized_criteria_list

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
        print(f"Warning: No rubric found for prompt ID {prompt_id}, using fallback")
        return "Evaluate the response based on:\n- Relevance to the prompt (weight: 1)\n- Accuracy of information (weight: 1)\n- Clarity and coherence (weight: 1)"


def build_eval_prompt(prompt: str, rubric: str, response: str, verifier_prompt_template: str) -> List[Dict]:
    """Return conversation that asks a model to grade *response* under *rubric*."""
    formatted_prompt = verifier_prompt_template.format(
        prompt=prompt,
        response=response,
        rubric=rubric
    )
    return [
        {"role": "user", "content": formatted_prompt},
    ]


def parse_explicit_verifier_output(
    verifier_output: str,
    criteria_by_id: Dict[str, List[Dict]],
    prompt_id: str,
    use_weights: bool = True,
) -> tuple[int, str, Dict, bool]:
    """Parse the explicit verifier output and calculate weighted score.

    Returns:
        tuple: (weighted_score, debug_info, parsed_answers, parse_success)
    """
    try:
        import re

        cleaned_output = verifier_output.strip()

        # Method 1: Try to extract JSON object directly
        json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Method 2: Look for individual key-value pairs and reconstruct JSON
            json_pattern = re.findall(r'"(c\d+)"\s*:\s*"(yes|no)"', cleaned_output, re.IGNORECASE)
            if json_pattern:
                json_str = "{" + ", ".join([f'"{k}":"{v}"' for k, v in json_pattern]) + "}"
            else:
                return 0, f"No parseable JSON found in output: {verifier_output[:200]}...", {}, False

        # Parse the JSON
        try:
            answers = json.loads(json_str)
        except json.JSONDecodeError as e:
            try:
                pattern = r'"(c\d+)"\s*:\s*"(yes|no)"'
                matches = re.findall(pattern, json_str, re.IGNORECASE)
                if matches:
                    answers = {key: value.lower() for key, value in matches}
                else:
                    cleaned_json = json_str.replace('\\"', '"').replace('""', '"').replace("'", '"')
                    answers = json.loads(cleaned_json)
            except (json.JSONDecodeError, Exception):
                return 0, f"Invalid JSON after all attempts: {json_str[:500]}... | Error: {e}", {}, False

        if prompt_id not in criteria_by_id:
            return 0, f"No criteria found for prompt {prompt_id}", {}, False

        criteria_list = criteria_by_id[prompt_id]

        weights_by_id = {}
        criteria_text_by_id = {}
        for criterion_data in criteria_list:
            local_id = criterion_data.get("local_id", f"c{criteria_list.index(criterion_data)+1}")
            weight = criterion_data.get("weight", 1)
            criterion_text = criterion_data["criterion"]
            weights_by_id[local_id] = weight
            criteria_text_by_id[local_id] = criterion_text

        total_score = 0
        total_weight = 0
        debug_parts = []
        parsed_answers = {}

        expected_criteria = set(weights_by_id.keys())
        found_criteria = set(answers.keys())

        for criterion_id, answer in answers.items():
            weight = weights_by_id.get(criterion_id, 1)
            effective_weight = weight if use_weights else 1
            criterion_text = criteria_text_by_id.get(criterion_id, "Unknown criterion")

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
                score_contribution = effective_weight if answer.lower() == "yes" else 0
                total_score += score_contribution
                total_weight += effective_weight
                debug_parts.append(f"{criterion_id}:{answer}(w={effective_weight})")

        missing_criteria = expected_criteria - found_criteria
        if missing_criteria:
            for missing_id in missing_criteria:
                weight = weights_by_id[missing_id]
                effective_weight = weight if use_weights else 1
                criterion_text = criteria_text_by_id[missing_id]
                total_weight += effective_weight
                debug_parts.append(f"{missing_id}:MISSING(w={effective_weight})")

                parsed_answers[missing_id] = {
                    "answer": "no",
                    "weight": weight,
                    "effective_weight": effective_weight,
                    "criterion_text": criterion_text,
                    "found": False
                }

        debug_info = ""
        if missing_criteria:
            debug_info = f"Missing criteria: {list(missing_criteria)}"

        return total_score, debug_info, parsed_answers, True

    except Exception as e:
        return 0, f"Error parsing output: {e} | Raw output: {verifier_output[:500]}...", {}, False


def prepare_all_scoring_tasks(
    responses_data: List[Dict],
    rubrics_data: Dict[str, str],
    criteria_by_id: Dict[str, List[Dict]],
    verifier_prompt_template: str,
    verifier: str,
    responses_per_prompt: int = None,
    max_retries: int = 2,
    retry_temperature: float = 1.0,
    use_weights: bool = True,
    base_url: Optional[str] = None,
) -> List[tuple]:
    """Prepare all scoring tasks from all prompts for parallel processing."""
    all_tasks = []

    for prompt_data in responses_data:
        prompt_id = prompt_data["id"]
        prompt = prompt_data["prompt"]
        rubric = extract_rubric_from_loaded_data(rubrics_data, prompt_id)
        responses = prompt_data["responses"]

        if responses_per_prompt is not None and responses_per_prompt < len(responses):
            responses = responses[:responses_per_prompt]

        for response_idx, response in enumerate(responses):
            task = (
                prompt, rubric, response, response_idx,
                verifier_prompt_template, prompt_id, criteria_by_id,
                max_retries, retry_temperature, verifier, use_weights, base_url,
            )
            all_tasks.append(task)

    return all_tasks


def score_single_response_with_prompt_id(args_tuple) -> Dict:
    """Score a single response using the rubric. Includes prompt_id for grouping results."""
    (
        prompt, rubric, response, response_idx,
        verifier_prompt_template, prompt_id, criteria_by_id,
        max_retries, retry_temperature, verifier, use_weights, base_url,
    ) = args_tuple

    retry_attempts = []

    for attempt in range(max_retries + 1):
        try:
            conv = build_eval_prompt(prompt, rubric, response, verifier_prompt_template)
            current_temperature = 0.0 if attempt == 0 else retry_temperature

            score_text = generate_via_api(
                verifier,
                conv,
                max_tokens=None,
                temperature=current_temperature,
                base_url=base_url,
            )

            score, debug_info, parsed_answers, parse_success = parse_explicit_verifier_output(
                score_text, criteria_by_id, prompt_id, use_weights=use_weights,
            )

            retry_attempts.append({
                "attempt": attempt + 1,
                "temperature": current_temperature,
                "parse_success": parse_success,
                "score": score,
                "debug_info": debug_info
            })

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

            retry_attempts.append({
                "attempt": attempt + 1,
                "temperature": 0.0 if attempt == 0 else retry_temperature,
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


def score_responses_with_api(all_scoring_tasks: List[tuple], workers: int) -> List[Dict]:
    """Score all responses using API-based parallel processing."""
    scored_responses = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_single_response_with_prompt_id, task) for task in all_scoring_tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring responses"):
            scored_responses.append(future.result())
    return scored_responses


def score_responses_direct(
    responses_file: str,
    rubrics_file: str,
    output_file: str,
    verifier: str = "google/gemma-3-27b-it",
    workers: int = 16,
    max_retries: int = 2,
    retry_temperature: float = 1.0,
    sample_size: Optional[int] = None,
    responses_per_prompt: Optional[int] = None,
    filter_prompt_ids: Optional[List[str]] = None,
    use_weights: bool = True,
    base_url: Optional[str] = None,
) -> bool:
    """Score responses using rubrics via API.

    Args:
        responses_file: Path to responses JSON file
        rubrics_file: Path to rubrics JSON file
        output_file: Path to output JSON file
        verifier: Model name for scoring
        workers: Number of parallel workers
        max_retries: Maximum retry attempts
        retry_temperature: Temperature for retry attempts
        sample_size: Number of prompts to sample (None for all)
        responses_per_prompt: Number of responses per prompt to score (None for all)
        filter_prompt_ids: List of prompt IDs to filter (None for all)
        use_weights: Whether to apply criterion weights
        base_url: Override the default API base URL

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Scoring responses")
    print(f"  Responses file: {responses_file}")
    print(f"  Rubrics file: {rubrics_file}")
    print(f"  Output file: {output_file}")

    # Load the verifier prompt template
    verifier_prompt_template = load_verifier_prompt()

    # Load datasets
    responses_data = load_json_data(responses_file)
    rubrics_data, criteria_by_id = load_rubrics_from_file(rubrics_file)

    if sample_size:
        print(f"Sampling {sample_size} prompts...")
        responses_data = random.sample(responses_data, min(sample_size, len(responses_data)))

    if filter_prompt_ids:
        print(f"Filtering to {len(filter_prompt_ids)} specific prompt IDs...")
        responses_data = [
            prompt_data for prompt_data in responses_data
            if prompt_data["id"] in filter_prompt_ids
        ]

    print(f"Processing {len(responses_data)} prompts")

    rubrics_summary = collect_rubrics_summary(responses_data, rubrics_data)
    print(f"Found {rubrics_summary['total_unique_rubrics']} unique rubrics")

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
        base_url=base_url,
    )

    total_tasks = len(all_scoring_tasks)
    print(f"Total responses to score: {total_tasks}")

    scored_responses = score_responses_with_api(all_scoring_tasks, workers)

    # Calculate retry statistics
    total_final_successes = sum(1 for resp in scored_responses if resp.get("_parse_success", False))
    total_responses = len(scored_responses)
    initial_failures = sum(1 for resp in scored_responses if len(resp.get("retry_attempts", [])) > 1)
    retry_recoveries = sum(1 for resp in scored_responses if len(resp.get("retry_attempts", [])) > 1 and resp.get("_parse_success", False))

    print(f"Retry statistics:")
    print(f"  Total responses: {total_responses}")
    print(f"  Initial failures: {initial_failures}")
    print(f"  Retry recoveries: {retry_recoveries}")
    if total_responses > 0:
        print(f"  Final success rate: {(total_final_successes / total_responses * 100):.1f}%")
    else:
        print(f"  Final success rate: N/A (no responses to score)")

    # Group results by prompt
    results_by_prompt = {}
    for scored_response in scored_responses:
        prompt_id = scored_response["prompt_id"]
        results_by_prompt.setdefault(prompt_id, []).append(scored_response)

    # Create final results with prompt information
    final_results = []
    total_parse_failures = 0
    for prompt_data in responses_data:
        prompt_id = prompt_data["id"]
        prompt = prompt_data["prompt"]
        rubric = extract_rubric_from_loaded_data(rubrics_data, prompt_id)

        prompt_scored_responses = results_by_prompt.get(prompt_id, [])
        if not prompt_scored_responses:
            continue

        prompt_scored_responses.sort(key=lambda x: x["response_idx"])

        prompt_parse_failures = sum(1 for resp in prompt_scored_responses if not resp.get("_parse_success", False))
        total_parse_failures += prompt_parse_failures

        for resp in prompt_scored_responses:
            resp.pop("_parse_success", None)

        # Calculate criterion performance summary
        criterion_summary = {}
        sample_parsed_answers = prompt_scored_responses[0].get("parsed_answers", {})
        for criterion_id, criterion_data in sample_parsed_answers.items():
            criterion_text = criterion_data.get("criterion_text", "Unknown")
            weight = criterion_data.get("weight", 1)

            yes_count = sum(
                1 for resp in prompt_scored_responses
                if resp.get("parsed_answers", {}).get(criterion_id, {}).get("answer") == "yes"
            )
            no_count = len(prompt_scored_responses) - yes_count

            criterion_summary[criterion_id] = {
                "criterion_text": criterion_text,
                "weight": weight,
                "yes_count": yes_count,
                "no_count": no_count,
                "yes_percentage": (yes_count / len(prompt_scored_responses)) * 100 if prompt_scored_responses else 0
            }

        final_results.append({
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
        })

    # Save results
    output_data = final_results

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(output_data, fp, indent=2)

    print(f"Saved results for {len(final_results)} prompts to {output_file}")
    print(f"Total responses scored: {len(scored_responses)}")
    print(f"Total parse failures: {total_parse_failures}")

    return True


def score_responses(
    responses_file: str,
    rubrics_file: str,
    output_file: str,
    sample_size: Optional[int] = None,
    verifier: str = "google/gemma-3-27b-it",
    max_retries: int = 2,
    retry_temperature: float = 1.0,
    use_weights: bool = True,
    base_url: Optional[str] = None,
    workers: int = 128,
) -> bool:
    """Score responses using rubrics.

    Args:
        responses_file: Path to responses JSON file
        rubrics_file: Path to rubrics JSON file
        output_file: Path to output JSON file
        sample_size: Number of prompts to sample (None for all)
        verifier: Model name for scoring
        max_retries: Maximum retry attempts
        retry_temperature: Temperature for retry attempts
        use_weights: Whether to apply criterion weights
        base_url: Override the default API base URL
        workers: Number of parallel workers

    Returns:
        bool: True if successful, False otherwise
    """
    return score_responses_direct(
        responses_file=responses_file,
        rubrics_file=rubrics_file,
        output_file=output_file,
        verifier=verifier,
        workers=workers,
        max_retries=max_retries,
        retry_temperature=retry_temperature,
        sample_size=sample_size,
        responses_per_prompt=None,
        filter_prompt_ids=None,
        use_weights=use_weights,
        base_url=base_url,
    )
