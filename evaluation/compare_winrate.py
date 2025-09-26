#!/usr/bin/env python3
"""
compare_winrate_unified.py

Unified pipeline to:
  1) Generate responses for prompts taken from a reference responses JSON
  2) Compare the generated responses against the reference using an LLM judge

Inputs:
  - --model-id: model checkpoint or API model name for generation
  - --reference-file: path to reference responses JSON (e.g., data/generalist/ots_test_1k_qwen3_8b.json)

This script orchestrates generation (via generator/rollouts.py) and comparison
(via evaluation/judge_win_rates.py) so you only need to provide the model and
the reference file.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import re

# Ensure project root is on sys.path for package-style imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import generation API
from generator.rollouts import generate_responses  # type: ignore

# Import judge workflow components
from evaluation.judge_win_rates import Config as JudgeConfig, ComparisonWorkflow  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses and compare against a reference set to compute win rate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required inputs
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier. For vLLM: local/HF model path; for API: API model name",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        required=True,
        help="Path to reference responses JSON (prompts are extracted from this file)",
    )

    # Generation settings
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Limit number of prompts to process (None = all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Generation top-p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (0 or negative = provider default)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=128,
        help="Number of parallel workers for API generation and judge threads",
    )
    # Always generate exactly one response per prompt, sequential selection when sampling

    # Output locations
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated",
        help="Directory to save generated responses",
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Custom filename for generated responses (stored under --output-dir)",
    )

    # Judge settings
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-4.1",
        help="Judge model for pairwise comparison",
    )

    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Judge temperature",
    )
    # Judge max tokens removed; default behavior is used

    return parser.parse_args()


def sanitize_model_name_for_filename(model_id: str) -> str:
    name = model_id.split("/")[-1]
    name = name.replace("/", "_").replace(":", "_").replace("-", "_")
    return name


def build_generated_output_path(output_dir: str, output_filename: Optional[str], model_id: str, reference_file: str, temperature: float, top_p: float, num_prompts: Optional[int]) -> str:
    if output_filename:
        return str(Path(output_dir) / output_filename)
    model_name = sanitize_model_name_for_filename(model_id)
    base_label = Path(reference_file).stem
    prompts_part = f"{num_prompts}" if num_prompts is not None else "all"
    # Always 1 response per prompt in unified pipeline
    fname = f"Policy_Model_{model_name}_Temperature_{temperature}_TopP_{top_p}_{prompts_part}_Prompts_1_Responses_{base_label}.json"
    return str(Path(output_dir) / fname)


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def strip_think_blocks(text: str) -> str:
    """Remove any '<think>...</think>' blocks from a string (case-insensitive)."""
    if not isinstance(text, str):
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_entry_inplace(entry: dict) -> None:
    """Clean a single item dict in-place: strip think blocks from response fields."""
    if not isinstance(entry, dict):
        return
    # Common single response field
    if "response" in entry and isinstance(entry["response"], str):
        entry["response"] = strip_think_blocks(entry["response"]) 
    # Responses list
    if "responses" in entry and isinstance(entry["responses"], list):
        new_list = []
        for elem in entry["responses"]:
            if isinstance(elem, str):
                new_list.append(strip_think_blocks(elem))
            elif isinstance(elem, dict):
                # If nested dict holds a response string
                if "response" in elem and isinstance(elem["response"], str):
                    elem = dict(elem)
                    elem["response"] = strip_think_blocks(elem["response"]) 
                new_list.append(elem)
            else:
                new_list.append(elem)
        entry["responses"] = new_list


def build_cleaned_reference(reference_file: str) -> str:
    """Create a cleaned copy of the reference JSON with <think> blocks removed.

    Returns path to the cleaned file.
    """
    with open(reference_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Preserve top-level structure
    if isinstance(data, list):
        cleaned = []
        for item in data:
            if isinstance(item, dict):
                item_copy = dict(item)
                _clean_entry_inplace(item_copy)
                cleaned.append(item_copy)
            else:
                cleaned.append(item)
    elif isinstance(data, dict):
        cleaned = dict(data)
        # If nested list containers like 'data' or 'items'
        for key in ("responses", "data", "items"):
            if key in cleaned and isinstance(cleaned[key], list):
                new_list = []
                for item in cleaned[key]:
                    if isinstance(item, dict):
                        item_copy = dict(item)
                        _clean_entry_inplace(item_copy)
                        new_list.append(item_copy)
                    else:
                        new_list.append(item)
                cleaned[key] = new_list
        # Also clean direct single response fields if present on the root
        _clean_entry_inplace(cleaned)
    else:
        cleaned = data

    cleaned_path = str(Path(reference_file).with_name(Path(reference_file).stem + "_cleaned.json"))
    ensure_parent_dir(cleaned_path)
    with open(cleaned_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    return cleaned_path


def main() -> int:
    load_dotenv()
    args = parse_args()

    reference_file = args.reference_file
    if not Path(reference_file).exists():
        print(f"Error: reference file not found: {reference_file}")
        return 1

    # Decide generated output file path
    generated_output = build_generated_output_path(
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        model_id=args.model_id,
        reference_file=reference_file,
        temperature=args.temperature,
        top_p=args.top_p,
        num_prompts=args.num_prompts,
    )

    # Step 1: Generate responses (always fresh generation, one response per prompt, vLLM/local checkpoint)
    print("=== Generation Stage ===")
    print(f"Model: {args.model_id}")
    print(f"Reference prompts: {reference_file}")
    print(f"Saving generated responses to: {generated_output}")
    ensure_parent_dir(generated_output)

    ok = generate_responses(
        input_file=reference_file,
        output_file=generated_output,
        num_responses=1,
        sample_size=args.num_prompts,
        model_id=args.model_id,
        tensor_parallel_size=None,  # let rollouts decide based on GPUs
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sequential=True,  # deterministic selection when sampling
        use_api=False,    # always local/vLLM in this unified script
        api_max_retries=3,
        num_workers=args.num_workers,
        existing_file=None,
    )
    if not ok:
        print("Generation failed; aborting.")
        return 1

    # Optional: verify generated file structure is loadable
    try:
        with open(generated_output, "r", encoding="utf-8") as fp:
            _ = json.load(fp)
    except Exception as e:
        print(f"Error: generated file is not valid JSON: {generated_output}\n{e}")
        return 1

    # Step 2: Build a cleaned reference with think blocks removed
    cleaned_reference = build_cleaned_reference(reference_file)

    # Step 3: Compare generated (File A) vs cleaned reference (File B)
    print("\n=== Comparison Stage ===")
    judge_cfg = JudgeConfig(
        # Model settings
        judge_model=args.judge_model,
        max_tokens=None,
        temperature=args.judge_temperature,
        # Execution settings
        num_workers=args.num_workers,
        random_seed=42,
        sample_size=args.num_prompts,
        # File paths
        responses_a_path=generated_output,
        responses_b_path=cleaned_reference,
        comparison_template_path="evaluation/prompts/compare_2_responses.txt",
        output_path="",  # will be computed by workflow's save method if empty handled; supply explicit below
    )

    # The judge module expects an explicit output path; mirror its default naming
    from evaluation.judge_win_rates import generate_output_filename  # type: ignore
    judge_output = generate_output_filename(
        args.judge_model, generated_output, cleaned_reference, args.num_prompts
    )
    judge_cfg.output_path = judge_output

    judge_cfg.print_config()
    workflow = ComparisonWorkflow(judge_cfg)
    workflow.run()

    print("\nUnified compare-winrate pipeline completed successfully!")
    print(f"Generated: {generated_output}")
    print(f"Comparison: {judge_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


