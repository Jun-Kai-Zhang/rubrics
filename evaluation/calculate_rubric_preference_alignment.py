#!/usr/bin/env python3
"""calculate_rubric_preference_alignment.py

This script calculates how well rubric-based scores align with GPT-4.1 preferences.
It:
1. Uses GPT-4.1 to compare response pairs (get preference ground truth)
2. Uses rubrics to score both responses multiple times (default: 3) with majority voting
3. Compares scores to see if they align with preferences
4. Calculates accuracy:
   - 1.0 for correct alignment
   - 0.5 for ties (including cases where no majority is reached in voting)
   - 0.0 for misalignment
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_via_api, extract_final_answer_from_box


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Configuration settings for the alignment calculation."""
    # Model settings
    judge_model: str = "openai/gpt-4.1"
    scorer_model: str = "openai/gpt-4.1-mini"
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    
    # Execution settings
    num_workers: int = 64
    random_seed: int = 42
    sample_size: Optional[int] = None
    num_scoring_runs: int = 3
    
    # File paths
    responses_file: str = "common_subsets/health_1k_qwen_3_base.json"
    rubrics_file: str = "common_subsets/health_common_subset_medical_public_gemini_flash_lite_single.json"
    comparison_template_path: str = "evaluation/prompts/compare_2_responses.txt"
    verifier_template_path: str = "generator/prompts/verifier_explicit.txt"
    output_path: str = "evaluation/rubric_preference_alignment_results.json"
    cache_path: str = "evaluation/gpt4_preference_cache.json"


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------
def strip_think_blocks(text: str) -> str:
    """Remove any '<think>...</think>' blocks from a string (case-insensitive).
    
    Args:
        text: The text to clean
        
    Returns:
        The text with think blocks removed
    """
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------
def create_cache_key(prompt_id: str, response1: str, response2: str) -> str:
    """Create a deterministic cache key for a response pair.
    
    The key is order-independent to handle random flipping.
    """
    # Sort responses to ensure consistent ordering
    responses = sorted([response1, response2])
    
    # Create a string representation
    cache_str = f"{prompt_id}|{responses[0]}|{responses[1]}"
    
    # Hash it to keep the key manageable
    return hashlib.sha256(cache_str.encode('utf-8')).hexdigest()


def load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    """Load preference cache from file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from {cache_path}: {e}")
    return {}


def save_cache(cache: Dict[str, Dict[str, Any]], cache_path: str) -> None:
    """Save preference cache to file."""
    try:
        # Ensure directory exists
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_json(file_path: str) -> Any:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_template(file_path: str) -> str:
    """Load text template from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_responses(file_path: str) -> List[Dict[str, Any]]:
    """Load responses from file."""
    data = load_json(file_path)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'responses' in data:
        return data['responses']
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def load_rubrics(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load rubrics and return as a dict mapping id to rubric."""
    data = load_json(file_path)
    
    rubrics_list = data.get('rubrics', [])
    rubrics_dict = {}
    
    for rubric in rubrics_list:
        prompt_id = rubric.get('id')
        if prompt_id:
            rubrics_dict[prompt_id] = rubric
    
    return rubrics_dict


# ---------------------------------------------------------------------------
# GPT-4.1 Preference Comparison
# ---------------------------------------------------------------------------
def get_gpt4_preference(
    prompt_id: str,
    prompt: str,
    response1: str, 
    response2: str,
    template: str,
    config: Config,
    cache: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], bool]:
    """Get GPT-4.1's preference between two responses.
    
    Returns:
        Tuple of (result_dict, was_cached)
    """
    
    # Check cache first
    cache_key = create_cache_key(prompt_id, response1, response2)
    if cache_key in cache:
        cached_result = cache[cache_key]
        # Need to determine which response is which based on the sorted order
        responses_sorted = sorted([response1, response2])
        if response1 == responses_sorted[0]:
            # response1 is first in sorted order, so cached preference is correct
            return cached_result, True
        else:
            # response2 is first in sorted order, so need to flip the preference
            flipped_result = cached_result.copy()
            if flipped_result.get('preference') in [1, 2]:
                flipped_result['preference'] = 3 - flipped_result['preference']  # Flip 1->2, 2->1
            return flipped_result, True
    
    # Not in cache, make API call
    # Randomly flip order to avoid bias
    flip_order = random.choice([True, False])
    
    if flip_order:
        shown_response1, shown_response2 = response2, response1
        mapping = {1: 2, 2: 1}  # Map shown preference back to original
    else:
        shown_response1, shown_response2 = response1, response2
        mapping = {1: 1, 2: 2}
    
    # Create comparison prompt
    prompt_text = template.format(
        prompt=prompt,
        response1=shown_response1,
        response2=shown_response2
    )
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        full_response = generate_via_api(
            model=config.judge_model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        # Extract preference from boxed answer
        boxed_answer = extract_final_answer_from_box(full_response)
        
        if boxed_answer in ['1', '2']:
            shown_preference = int(boxed_answer)
            actual_preference = mapping[shown_preference]
            
            result = {
                'success': True,
                'preference': actual_preference,
                'flip_order': flip_order,
                'full_response': full_response
            }
            
            # Store in cache with preference relative to sorted order
            responses_sorted = sorted([response1, response2])
            if response1 == responses_sorted[0]:
                cache_preference = actual_preference
            else:
                cache_preference = 3 - actual_preference  # Flip preference for cache
            
            cache[cache_key] = {
                'success': True,
                'preference': cache_preference,
                'flip_order': flip_order,
                'full_response': full_response
            }
            
            return result, False
        else:
            result = {
                'success': False,
                'error': 'Could not extract preference',
                'full_response': full_response
            }
            return result, False
            
    except Exception as e:
        result = {
            'success': False,
            'error': str(e)
        }
        return result, False


# ---------------------------------------------------------------------------
# Rubric-based Scoring
# ---------------------------------------------------------------------------
def format_rubric_for_scoring(rubric: Dict[str, Any]) -> str:
    """Format rubric into a string for the scoring prompt."""
    criteria_text = []
    for criterion in rubric.get('criteria', []):
        local_id = criterion.get('local_id', '')
        text = criterion.get('criterion', '')
        weight = criterion.get('weight')
        criteria_text.append(f"{local_id}: {text} (Weight: {weight})")
    
    return "\n".join(criteria_text)


def parse_scoring_response(response: str) -> Dict[str, str]:
    """Parse the JSON response from the scorer."""
    # Try to extract JSON from the response
    import re
    
    # Find JSON-like content
    json_match = re.search(r'\{[^{}]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # If parsing fails, return empty dict
    return {}


def calculate_rubric_score(
    scoring_results: Dict[str, str],
    rubric: Dict[str, Any]
) -> float:
    """Calculate the total score based on scoring results and rubric weights."""
    total_score = 0
    total_weight = 0
    
    for criterion in rubric.get('criteria', []):
        local_id = criterion.get('local_id', '')
        weight = criterion.get('weight', 1)
        
        if local_id in scoring_results:
            if scoring_results[local_id].lower() == 'yes':
                total_score += weight
            total_weight += weight
    
    # Return normalized score (0-1)
    if total_weight > 0:
        return total_score / total_weight
    else:
        return 0.0


def score_response_with_rubric(
    prompt: str,
    response: str,
    rubric: Dict[str, Any],
    template: str,
    config: Config
) -> Dict[str, Any]:
    """Score a single response using the rubric."""
    
    rubric_text = format_rubric_for_scoring(rubric)
    
    # Create scoring prompt
    prompt_text = template.format(
        prompt=prompt,
        response=response,
        rubric=rubric_text
    )
    
    messages = [{"role": "user", "content": prompt_text}]
    
    try:
        full_response = generate_via_api(
            model=config.scorer_model,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        # Parse the scoring results
        scoring_results = parse_scoring_response(full_response)
        
        # Calculate score
        score = calculate_rubric_score(scoring_results, rubric)
        
        return {
            'success': True,
            'score': score,
            'scoring_results': scoring_results,
            'full_response': full_response
        }
        
    except Exception as e:
        return {
            'success': False,
            'score': 0.0,
            'error': str(e)
        }


def score_response_with_rubric_multiple_runs(
    prompt: str,
    response: str,
    rubric: Dict[str, Any],
    template: str,
    config: Config,
    num_runs: int = 3
) -> Dict[str, Any]:
    """Score a response multiple times and take majority vote (excluding ties).
    
    If no clear majority is reached (e.g., all scores are different or there's a 
    tie between multiple scores), the score will be None. These cases are treated 
    as ties (0.5) in the alignment calculation.
    
    Returns:
        Dict containing:
        - success: Whether scoring succeeded (True even if no majority)
        - score: The majority vote score (or None if no majority)
        - all_scores: List of all scores from successful runs
        - runs: List of all run results
        - voting_details: Details about the voting process
    """
    
    runs = []
    successful_scores = []
    
    # Run scoring multiple times
    for i in range(num_runs):
        result = score_response_with_rubric(prompt, response, rubric, template, config)
        runs.append(result)
        if result['success']:
            successful_scores.append(result['score'])
    
    if not successful_scores:
        return {
            'success': False,
            'score': None,
            'all_scores': [],
            'runs': runs,
            'error': 'All scoring attempts failed'
        }
    
    # Count occurrences of each score
    score_counts = {}
    for score in successful_scores:
        score_counts[score] = score_counts.get(score, 0) + 1
    
    # Find the score with the most votes (majority)
    max_count = max(score_counts.values())
    scores_with_max_count = [score for score, count in score_counts.items() if count == max_count]
    
    # Check if we have a clear majority (not a tie)
    if len(scores_with_max_count) == 1 and max_count > len(successful_scores) / 2:
        majority_score = scores_with_max_count[0]
    else:
        # No clear majority or tie between multiple scores
        majority_score = None
    
    return {
        'success': True if majority_score is not None else False,
        'score': majority_score,
        'all_scores': successful_scores,
        'runs': runs,
        'voting_details': {
            'score_counts': score_counts,
            'max_count': max_count,
            'scores_with_max_count': scores_with_max_count,
            'total_successful_runs': len(successful_scores),
            'has_majority': majority_score is not None
        }
    }


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------
def process_single_prompt(
    prompt_data: Dict[str, Any],
    rubric: Dict[str, Any],
    comparison_template: str,
    verifier_template: str,
    config: Config,
    cache: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], bool]:
    """Process a single prompt: get preference and scores.
    
    Returns:
        Tuple of (result_dict, preference_was_cached)
    """
    
    prompt_id = prompt_data.get('id')
    prompt = prompt_data.get('prompt')
    responses = prompt_data.get('responses', [])
    
    if len(responses) < 2:
        return {
            'id': prompt_id,
            'error': 'Not enough responses',
            'success': False
        }, False
    
    # Take first two responses
    response1, response2 = responses[0], responses[1]
    
    # Clean responses by removing think blocks
    response1 = strip_think_blocks(response1)
    response2 = strip_think_blocks(response2)
    
    # Get GPT-4.1 preference
    preference_result, was_cached = get_gpt4_preference(
        prompt_id, prompt, response1, response2, comparison_template, config, cache
    )
    
    # Score both responses with rubric using multiple runs
    num_scoring_runs = getattr(config, 'num_scoring_runs', 3)
    score1_result = score_response_with_rubric_multiple_runs(
        prompt, response1, rubric, verifier_template, config, num_scoring_runs
    )
    
    score2_result = score_response_with_rubric_multiple_runs(
        prompt, response2, rubric, verifier_template, config, num_scoring_runs
    )
    
    # Determine if scores align with preference
    alignment = None
    if preference_result['success'] and score1_result['success'] and score2_result['success']:
        score1 = score1_result['score']
        score2 = score2_result['score']
        preference = preference_result['preference']
        
        # If either score is None (no majority in voting), treat as tie
        if score1 is None or score2 is None:
            # No majority in rubric voting - treat as inability to distinguish (tie)
            alignment = 0.5
        elif score1 == score2:
            # Actual tie in scores - counts as 0.5
            alignment = 0.5
        elif (score1 > score2 and preference == 1) or (score2 > score1 and preference == 2):
            # Scores align with preference
            alignment = 1.0
        else:
            # Scores don't align with preference
            alignment = 0.0
    
    return {
        'id': prompt_id,
        'prompt': prompt,
        'preference': preference_result.get('preference'),
        'preference_success': preference_result['success'],
        'score1': score1_result.get('score'),
        'score2': score2_result.get('score'),
        'score1_success': score1_result['success'],
        'score2_success': score2_result['success'],
        'alignment': alignment,
        'details': {
            'preference_result': preference_result,
            'score1_result': score1_result,
            'score2_result': score2_result
        }
    }, was_cached


def calculate_alignment(config: Config) -> None:
    """Main function to calculate alignment between rubric scores and preferences."""
    
    # Load data
    print("Loading data...")
    responses = load_responses(config.responses_file)
    rubrics = load_rubrics(config.rubrics_file)
    comparison_template = load_template(config.comparison_template_path)
    verifier_template = load_template(config.verifier_template_path)
    
    # Load cache
    cache = load_cache(config.cache_path)
    print(f"Loaded cache with {len(cache)} entries")
    
    # Filter to prompts that have rubrics
    filtered_responses = [r for r in responses if r.get('id') in rubrics]
    print(f"Found {len(filtered_responses)} prompts with rubrics out of {len(responses)} total")
    
    # Apply sampling if needed
    if config.sample_size and config.sample_size < len(filtered_responses):
        random.seed(config.random_seed)
        filtered_responses = random.sample(filtered_responses, config.sample_size)
        print(f"Sampled {config.sample_size} prompts")
    
    # Process all prompts in parallel
    print(f"Processing {len(filtered_responses)} prompts with {config.num_workers} workers...")
    results = []
    cache_hits = 0
    
    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = []
        
        for prompt_data in filtered_responses:
            prompt_id = prompt_data.get('id')
            rubric = rubrics.get(prompt_id)
            
            if rubric:
                future = executor.submit(
                    process_single_prompt,
                    prompt_data,
                    rubric,
                    comparison_template,
                    verifier_template,
                    config,
                    cache
                )
                futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result, was_cached = future.result()
                results.append(result)
                if was_cached:
                    cache_hits += 1
            except Exception as e:
                print(f"Error processing prompt: {e}")
    
    print(f"Cache hits: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
    
    # Save updated cache
    save_cache(cache, config.cache_path)
    print(f"Saved cache with {len(cache)} entries")
    
    # Calculate overall statistics
    successful_results = [r for r in results if r.get('alignment') is not None]
    total_alignment = sum(r['alignment'] for r in successful_results)
    accuracy = total_alignment / len(successful_results) if successful_results else 0.0
    
    # Count different outcomes
    perfect_alignments = sum(1 for r in successful_results if r['alignment'] == 1.0)
    misalignments = sum(1 for r in successful_results if r['alignment'] == 0.0)
    ties = sum(1 for r in successful_results if r['alignment'] == 0.5)
    
    # Count actual score ties vs no-majority cases (both contribute 0.5 to alignment)
    actual_ties = sum(1 for r in successful_results 
                     if r['alignment'] == 0.5 and 
                        r['score1'] is not None and r['score2'] is not None and 
                        r['score1'] == r['score2'])
    no_majority_ties = sum(1 for r in successful_results 
                          if r['alignment'] == 0.5 and 
                             (r['score1'] is None or r['score2'] is None))
    
    # Save results
    output_data = {
        'metadata': {
            'config': {
                'judge_model': config.judge_model,
                'scorer_model': config.scorer_model,
                'responses_file': config.responses_file,
                'rubrics_file': config.rubrics_file,
                'sample_size': config.sample_size,
                'num_workers': config.num_workers,
                'num_scoring_runs': config.num_scoring_runs
            },
            'statistics': {
                'total_prompts': len(results),
                'successful_evaluations': len(successful_results),
                'failed_evaluations': len(results) - len(successful_results),
                'accuracy': accuracy,
                'perfect_alignments': perfect_alignments,
                'misalignments': misalignments,
                'ties': ties,
                'actual_score_ties': actual_ties,
                'no_majority_ties': no_majority_ties
            }
        },
        'results': results
    }
    
    # Ensure output directory exists
    output_dir = os.path.dirname(config.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(config.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    print(f"Total prompts processed: {len(results)}")
    print(f"Successful evaluations: {len(successful_results)}")
    print(f"Failed evaluations: {len(results) - len(successful_results)}")
    print(f"\nScoring Configuration:")
    print(f"  Number of scoring runs per response: {config.num_scoring_runs}")
    print(f"  Majority voting: Enabled (no-majority cases treated as ties)")
    print(f"\nAlignment Results:")
    print(f"  Perfect alignments: {perfect_alignments} ({perfect_alignments/len(successful_results)*100:.1f}%)")
    print(f"  Misalignments: {misalignments} ({misalignments/len(successful_results)*100:.1f}%)")
    print(f"  Ties (total): {ties} ({ties/len(successful_results)*100:.1f}%)")
    if ties > 0:
        print(f"    - Actual score ties: {actual_ties} ({actual_ties/len(successful_results)*100:.1f}%)")
        print(f"    - No majority cases: {no_majority_ties} ({no_majority_ties/len(successful_results)*100:.1f}%)")
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"\nResults saved to: {config.output_path}")


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate alignment between rubric-based scores and GPT-4.1 preferences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    model_group = parser.add_argument_group('Model Settings')
    model_group.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4.1",
        help="Model to use for preference comparison"
    )
    model_group.add_argument(
        "--scorer-model",
        type=str,
        default="openai/gpt-4.1-mini",
        help="Model to use for rubric-based scoring"
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for model responses"
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model responses"
    )
    
    # Execution settings
    exec_group = parser.add_argument_group('Execution Settings')
    exec_group.add_argument(
        "--workers",
        type=int,
        default=256,
        help="Number of parallel worker threads [[memory:6163808]]"
    )
    exec_group.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    exec_group.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of prompts to sample (None = use all)"
    )
    exec_group.add_argument(
        "--num-scoring-runs",
        type=int,
        default=1,
        help="Number of times to run rubric scoring for majority vote (default: 3)"
    )
    
    # File paths
    file_group = parser.add_argument_group('File Paths')
    file_group.add_argument(
        "--responses-file",
        type=str,
        default="common_subsets/health_1k_qwen_3_base.json",
        help="Path to responses JSON file"
    )
    file_group.add_argument(
        "--rubrics-file",
        type=str,
        default="common_subsets/health_common_subset_medical_public_gemini_flash_lite_single.json",
        help="Path to rubrics JSON file"
    )
    file_group.add_argument(
        "--comparison-template",
        type=str,
        default="evaluation/prompts/compare_2_responses.txt",
        help="Path to comparison prompt template"
    )
    file_group.add_argument(
        "--verifier-template",
        type=str,
        default="generator/prompts/verifier_explicit.txt",
        help="Path to verifier prompt template"
    )
    file_group.add_argument(
        "--output",
        type=str,
        default="evaluation/rubric_preference_alignment_results.json",
        help="Output file path"
    )
    file_group.add_argument(
        "--cache",
        type=str,
        default="evaluation/gpt4_preference_cache.json",
        help="Path to cache file for GPT-4.1 preferences"
    )
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Main entry point for the script."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    config = Config(
        judge_model=args.judge_model,
        scorer_model=args.scorer_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.workers,
        random_seed=args.random_seed,
        num_scoring_runs=args.num_scoring_runs,
        sample_size=args.sample_size,
        responses_file=args.responses_file,
        rubrics_file=args.rubrics_file,
        comparison_template_path=args.comparison_template,
        verifier_template_path=args.verifier_template,
        output_path=args.output,
        cache_path=args.cache
    )
    
    # Run calculation
    try:
        calculate_alignment(config)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
