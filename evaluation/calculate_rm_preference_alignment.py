#!/usr/bin/env python3
"""calculate_rm_preference_alignment.py

This script calculates how well reward-model scores align with GPT-4.1 preferences.
It mirrors `evaluation/calculate_rubric_preference_alignment.py` but replaces the
rubric-based scoring with a reward model (see `evaluation/score_responses_rm.py`).

It:
1. Uses GPT-4.1 to compare response pairs (get preference ground truth)
2. Uses a reward model to score both responses
3. Compares scores to see if they align with preferences
4. Calculates accuracy (0.5 for ties)

The script shares the same GPT-4.1 preference cache format/path as
`evaluation/calculate_rubric_preference_alignment.py`.
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
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path for imports (for utils)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add current directory to path for importing RM scorer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import generate_via_api, extract_final_answer_from_box

try:
    # Import reward model scorer utilities
    from score_responses_rm import (
        MultiGPURewardModelScorer,
        PromptResponseDataset,
        get_gpu_count,
        init_distributed,
        cleanup_distributed,
        is_main_process,
    )
except Exception:
    # Fallback minimal shims if direct import fails (should not happen)
    MultiGPURewardModelScorer = None  # type: ignore
    PromptResponseDataset = None  # type: ignore
    def get_gpu_count() -> int:  # type: ignore
        return 0
    def init_distributed(args: argparse.Namespace) -> bool:  # type: ignore
        return False
    def cleanup_distributed() -> None:  # type: ignore
        return None
    def is_main_process() -> bool:  # type: ignore
        return True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Configuration settings for the alignment calculation using RM."""
    # Preference (GPT judge) settings
    judge_model: str = "openai/gpt-4.1"
    max_tokens: Optional[int] = None
    temperature: float = 0.0

    # Execution settings
    num_workers: int = 64
    random_seed: int = 42
    sample_size: Optional[int] = None

    # File paths
    responses_file: str = "common_subsets/health_1k_qwen_3_base.json"
    comparison_template_path: str = "evaluation/prompts/compare_2_responses.txt"
    output_path: str = "evaluation/rm_preference_alignment_results.json"
    cache_path: str = "evaluation/gpt4_preference_cache.json"

    # Reward model settings
    rm_model: str = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    rm_no_flash_attention: bool = False
    rm_gpus: Optional[str] = None  # comma-separated GPU IDs
    rm_parallel_mode: str = "dp"  # "dp" or "ddp"
    rm_local_rank: int = -1
    rm_batch_size: int = 128  # per GPU
    rm_num_workers: int = 0  # dataloader workers per GPU
    rm_max_length: int = 2048


# ---------------------------------------------------------------------------
# Text Processing
# ---------------------------------------------------------------------------
def strip_think_blocks(text: str) -> str:
    """Remove any '<think>...</think>' blocks from a string (case-insensitive)."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Cache Management (shared format with rubric alignment script)
# ---------------------------------------------------------------------------
def create_cache_key(prompt_id: str, response1: str, response2: str) -> str:
    """Create a deterministic, order-independent cache key for a response pair."""
    responses = sorted([response1, response2])
    cache_str = f"{prompt_id}|{responses[0]}|{responses[1]}"
    return hashlib.sha256(cache_str.encode('utf-8')).hexdigest()


def load_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from {cache_path}: {e}")
    return {}


def save_cache(cache: Dict[str, Dict[str, Any]], cache_path: str) -> None:
    try:
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
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_template(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_responses(file_path: str) -> List[Dict[str, Any]]:
    data = load_json(file_path)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'responses' in data:
        return data['responses']
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


# ---------------------------------------------------------------------------
# GPT-4.1 Preference Comparison (shared cache)
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
    """Get GPT-4.1's preference between two responses, using shared cache.

    Returns:
        Tuple of (result_dict, was_cached)
    """
    cache_key = create_cache_key(prompt_id, response1, response2)
    if cache_key in cache:
        cached_result = cache[cache_key]
        responses_sorted = sorted([response1, response2])
        if response1 == responses_sorted[0]:
            return cached_result, True
        else:
            flipped = cached_result.copy()
            if flipped.get('preference') in [1, 2]:
                flipped['preference'] = 3 - flipped['preference']
            return flipped, True

    flip_order = random.choice([True, False])
    if flip_order:
        shown_response1, shown_response2 = response2, response1
        mapping = {1: 2, 2: 1}
    else:
        shown_response1, shown_response2 = response1, response2
        mapping = {1: 1, 2: 2}

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

            responses_sorted = sorted([response1, response2])
            if response1 == responses_sorted[0]:
                cache_preference = actual_preference
            else:
                cache_preference = 3 - actual_preference

            cache[cache_key] = {
                'success': True,
                'preference': cache_preference,
                'flip_order': flip_order,
                'full_response': full_response
            }
            return result, False
        else:
            return {
                'success': False,
                'error': 'Could not extract preference',
                'full_response': full_response
            }, False
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }, False


# ---------------------------------------------------------------------------
# Alignment Calculation using Reward Model
# ---------------------------------------------------------------------------
def build_pair_tasks(responses: List[Dict[str, Any]]) -> List[Tuple[str, str, str, int]]:
    """Build tasks for RM scoring: (prompt, response, prompt_id, response_idx)."""
    tasks: List[Tuple[str, str, str, int]] = []
    for item in responses:
        prompt_id = item.get('id')
        prompt = item.get('prompt')
        resp_list = item.get('responses', [])
        if not prompt_id or not prompt or len(resp_list) < 2:
            continue
        # Take first two, strip think blocks
        r1 = strip_think_blocks(resp_list[0])
        r2 = strip_think_blocks(resp_list[1])
        tasks.append((prompt, r1, prompt_id, 0))
        tasks.append((prompt, r2, prompt_id, 1))
    return tasks


def calculate_alignment(config: Config) -> None:
    """Main function to calculate alignment between RM scores and GPT-4.1 preferences."""
    # Load data
    print("Loading data...")
    responses = load_responses(config.responses_file)
    comparison_template = load_template(config.comparison_template_path)

    # Load cache
    cache = load_cache(config.cache_path)
    print(f"Loaded cache with {len(cache)} entries")

    # Filter prompts with at least two responses
    filtered = [r for r in responses if len(r.get('responses', [])) >= 2]
    print(f"Found {len(filtered)} prompts with >=2 responses out of {len(responses)} total")

    # Apply sampling if needed
    if config.sample_size and config.sample_size < len(filtered):
        random.seed(config.random_seed)
        filtered = random.sample(filtered, config.sample_size)
        print(f"Sampled {config.sample_size} prompts")

    # Step 1: Compute GPT-4.1 preferences (parallel, with cache)
    print(f"Computing preferences for {len(filtered)} prompts with {config.num_workers} workers...")
    preferences: Dict[str, Dict[str, Any]] = {}
    cache_hits = 0

    def _pref_worker(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], bool]:
        prompt_id = item.get('id')
        prompt = item.get('prompt')
        resp_list = item.get('responses', [])
        r1 = strip_think_blocks(resp_list[0])
        r2 = strip_think_blocks(resp_list[1])
        result, was_cached = get_gpt4_preference(
            prompt_id, prompt, r1, r2, comparison_template, config, cache
        )
        return prompt_id, result, was_cached

    with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
        futures = [executor.submit(_pref_worker, item) for item in filtered]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preferences"):
            try:
                pid, res, was_cached = fut.result()
                preferences[pid] = res
                if was_cached:
                    cache_hits += 1
            except Exception as e:
                print(f"Error computing preference: {e}")

    print(f"Cache hits: {cache_hits}/{len(filtered)} ({(cache_hits/max(len(filtered),1))*100:.1f}%)")

    # Step 2: Score with Reward Model (batched on GPU)
    if MultiGPURewardModelScorer is None or PromptResponseDataset is None:
        raise RuntimeError("Failed to import reward model scorer utilities from score_responses_rm.py")

    # Initialize (optional) distributed
    class ArgsShim:
        parallel_mode = config.rm_parallel_mode
        local_rank = config.rm_local_rank
    is_distributed = init_distributed(ArgsShim())

    # Parse GPU IDs
    gpu_ids = None
    if config.rm_gpus:
        gpu_ids = [int(x.strip()) for x in config.rm_gpus.split(',') if x.strip()]

    if is_main_process():
        print("\nInitializing Reward Model...")
        print(f"Model: {config.rm_model}")
        print(f"Parallel mode: {config.rm_parallel_mode.upper()}")
        print(f"Batch size per GPU: {config.rm_batch_size}")
        print(f"Max sequence length: {config.rm_max_length}")

    try:
        scorer = MultiGPURewardModelScorer(
            model_name=config.rm_model,
            gpu_ids=gpu_ids,
            use_flash_attention=not config.rm_no_flash_attention,
            parallel_mode=config.rm_parallel_mode,
            local_rank=config.rm_local_rank,
            max_length=config.rm_max_length,
        )
    except Exception as e:
        cleanup_distributed()
        raise

    # Build RM tasks for the first two responses per prompt
    tasks = build_pair_tasks(filtered)
    total_tasks = len(tasks)
    if is_main_process():
        effective_bs = config.rm_batch_size * (len(gpu_ids) if gpu_ids else get_gpu_count())
        print(f"Scoring {total_tasks} responses with effective batch size {effective_bs}")

    # Create dataset and dataloader
    try:
        from torch.utils.data import DataLoader
    except Exception:
        cleanup_distributed()
        raise

    dataset = PromptResponseDataset(tasks)
    dataloader = DataLoader(
        dataset,
        batch_size=config.rm_batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.rm_num_workers,
        collate_fn=scorer.collate_fn,
        pin_memory=True if config.rm_num_workers > 0 else False,
    )

    try:
        scored_results = scorer.score_dataloader(dataloader, total_tasks)
    except Exception as e:
        cleanup_distributed()
        raise

    # Only proceed on main process for result aggregation
    if not is_main_process():
        cleanup_distributed()
        return

    # Organize scores as {prompt_id: {0: score1, 1: score2}}
    scores_by_prompt: Dict[str, Dict[int, float]] = {}
    for r in scored_results:
        pid = r["prompt_id"]
        ridx = r["response_idx"]
        score = r["score"]
        scores_by_prompt.setdefault(pid, {})[ridx] = float(score)

    # Step 3: Compute alignment
    results: List[Dict[str, Any]] = []
    successful_alignments = 0
    perfect_alignments = 0
    misalignments = 0
    ties = 0

    for item in filtered:
        pid = item.get('id')
        prompt = item.get('prompt')
        pref = preferences.get(pid, {"success": False})
        prompt_scores = scores_by_prompt.get(pid, {})

        alignment = None
        score1 = prompt_scores.get(0)
        score2 = prompt_scores.get(1)

        if pref.get('success') and score1 is not None and score2 is not None:
            if score1 == score2:
                alignment = 0.5
                ties += 1
            else:
                preferred = pref.get('preference')
                if (score1 > score2 and preferred == 1) or (score2 > score1 and preferred == 2):
                    alignment = 1.0
                    perfect_alignments += 1
                else:
                    alignment = 0.0
                    misalignments += 1
            successful_alignments += 1

        results.append({
            'id': pid,
            'prompt': prompt,
            'preference': pref.get('preference'),
            'preference_success': bool(pref.get('success')),
            'score1': score1,
            'score2': score2,
            'alignment': alignment,
            'details': {
                'preference_result': pref,
            }
        })

    # Save updated cache
    save_cache(cache, config.cache_path)
    print(f"Saved cache with {len(cache)} entries")

    accuracy = (sum(r['alignment'] for r in results if r.get('alignment') is not None) / successful_alignments) if successful_alignments else 0.0

    # Save results
    output_data = {
        'metadata': {
            'config': {
                'judge_model': config.judge_model,
                'rm_model': config.rm_model,
                'responses_file': config.responses_file,
                'sample_size': config.sample_size,
                'num_workers': config.num_workers,
                'rm_batch_size': config.rm_batch_size,
                'rm_gpus': config.rm_gpus,
                'rm_parallel_mode': config.rm_parallel_mode,
                'rm_max_length': config.rm_max_length,
            },
            'statistics': {
                'total_prompts': len(filtered),
                'successful_evaluations': successful_alignments,
                'failed_evaluations': len(filtered) - successful_alignments,
                'accuracy': accuracy,
                'perfect_alignments': perfect_alignments,
                'misalignments': misalignments,
                'ties': ties,
            }
        },
        'results': results
    }

    output_dir = os.path.dirname(config.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(config.output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY (Reward Model vs GPT-4.1 Preferences)")
    print("="*60)
    print(f"Total prompts processed: {len(filtered)}")
    print(f"Successful evaluations: {successful_alignments}")
    print(f"Failed evaluations: {len(filtered) - successful_alignments}")
    print(f"\nAlignment Results:")
    if successful_alignments:
        print(f"  Perfect alignments: {perfect_alignments} ({perfect_alignments/successful_alignments*100:.1f}%)")
        print(f"  Misalignments: {misalignments} ({misalignments/successful_alignments*100:.1f}%)")
        print(f"  Ties: {ties} ({ties/successful_alignments*100:.1f}%)")
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"\nResults saved to: {config.output_path}")

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate alignment between reward-model scores and GPT-4.1 preferences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Preference model (GPT judge) settings
    judge_group = parser.add_argument_group('Preference (GPT Judge) Settings')
    judge_group.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4.1",
        help="Model to use for preference comparison"
    )
    judge_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for judge model responses"
    )
    judge_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for judge model responses"
    )

    # Execution settings
    exec_group = parser.add_argument_group('Execution Settings')
    exec_group.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of parallel worker threads"
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

    # File paths
    file_group = parser.add_argument_group('File Paths')
    file_group.add_argument(
        "--responses-file",
        type=str,
        default="common_subsets/health_1k_qwen_3_base.json",
        help="Path to responses JSON file"
    )
    file_group.add_argument(
        "--comparison-template",
        type=str,
        default="evaluation/prompts/compare_2_responses.txt",
        help="Path to comparison prompt template"
    )
    file_group.add_argument(
        "--output",
        type=str,
        default="evaluation/rm_preference_alignment_results.json",
        help="Output file path"
    )
    file_group.add_argument(
        "--cache",
        type=str,
        default="evaluation/gpt4_preference_cache.json",
        help="Path to cache file for GPT-4.1 preferences (shared)"
    )

    # Reward model configuration
    rm_group = parser.add_argument_group('Reward Model Configuration')
    rm_group.add_argument(
        "--model",
        type=str,
        default="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
        help="Reward model to use for scoring"
    )
    rm_group.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable Flash Attention 2 (use if not available)"
    )
    rm_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs (e.g., '0,1,2'). Default: all GPUs"
    )
    rm_group.add_argument(
        "--parallel-mode",
        type=str,
        choices=["dp", "ddp"],
        default="dp",
        help="Parallelization mode: 'dp' or 'ddp'"
    )
    rm_group.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    rm_group.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size per GPU for inference"
    )
    rm_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers per GPU [[memory:6163808]]"
    )
    rm_group.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    load_dotenv()
    args = parse_arguments()

    config = Config(
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_workers=args.workers,
        random_seed=args.random_seed,
        sample_size=args.sample_size,
        responses_file=args.responses_file,
        comparison_template_path=args.comparison_template,
        output_path=args.output,
        cache_path=args.cache,
        rm_model=args.model,
        rm_no_flash_attention=args.no_flash_attention,
        rm_gpus=args.gpus,
        rm_parallel_mode=args.parallel_mode,
        rm_local_rank=args.local_rank,
        rm_batch_size=args.batch_size,
        rm_num_workers=args.num_workers,
        rm_max_length=args.max_length,
    )

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


