"""Generate responses for prompts using an OpenAI-compatible API.

Supports multi-model generation, resume from existing results, and
periodic checkpointing.  All model calls go through ``generate_via_api``
(litellm proxy, vLLM server, etc.).
"""

import json
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from generator.utils import generate_via_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt_as_conversation(prompt_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt_text}]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses using an OpenAI-compatible API"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Model identifier(s). Can specify multiple models.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Number of prompts to process",
    )
    parser.add_argument(
        "--num-responses-per-model", type=int, default=1,
        help="Responses per model per prompt (default: 1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=0,
        help="Max tokens to generate (0 = provider default)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--output-filename", type=str, default=None,
        help="Custom output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="Prompts JSON file (list of {\"prompt\": ..., \"id\": ...})",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use sequential loading instead of random sampling",
    )
    parser.add_argument(
        "--api-max-retries", type=int, default=3,
        help="Max API retries (default: 3)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=16,
        help="Parallel workers for API calls (default: 16)",
    )
    parser.add_argument(
        "--existing-file", type=str, default=None,
        help="Existing results JSON for resume support",
    )
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="Base URL for the OpenAI-compatible API (default: from env/config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

SOTA_MODELS = [
    "openai/gpt-5",
    "openai/gpt-4o-2024-05-13",
    "openai/gpt-4.1",
    "openai/o3",
    "openai/o1-2024-12-17",
    "openai/o4-mini",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "fireworks_ai/deepseek-v3",
    "fireworks_ai/deepseek-r1",
    "fireworks_ai/kimi-k2-instruct",
    "fireworks_ai/glm-4p5",
    "fireworks_ai/qwen3-235b-a22b-instruct-2507",
    "mistral/mistral-medium-latest",
]


def generate_responses(
    input_file: str,
    output_file: str,
    num_responses_per_model: int,
    sample_size: Optional[int],
    model_ids: List[str],
    max_tokens: int = 0,
    temperature: float = 1.0,
    top_p: float = 0.95,
    sequential: bool = False,
    api_max_retries: int = 3,
    num_workers: int = 64,
    existing_file: Optional[str] = None,
    base_url: Optional[str] = None,
) -> bool:
    """Generate responses for prompts using an OpenAI-compatible API.

    Returns True on success.
    """
    random.seed(42)

    # ------------------------------------------------------------------
    # Load prompts
    # ------------------------------------------------------------------
    prompts_data = _load_prompts(input_file, sample_size, sequential)
    actual_num_prompts = len(prompts_data)
    print(f"Loaded {actual_num_prompts} prompts")

    # ------------------------------------------------------------------
    # Load existing results for resume
    # ------------------------------------------------------------------
    existing_by_prompt_and_model: Dict[str, Dict[str, List]] = {}
    if existing_file and os.path.exists(existing_file):
        try:
            with open(existing_file, "r", encoding="utf-8") as efp:
                existing_data = json.load(efp)
            if isinstance(existing_data, list):
                for entry in existing_data:
                    pid = entry.get("id") or entry.get("task_id")
                    if not pid:
                        continue
                    by_model: Dict[str, List] = {}
                    for resp in entry.get("responses", []):
                        if isinstance(resp, dict) and "generator" in resp:
                            by_model.setdefault(resp["generator"], []).append(resp)
                        else:
                            by_model.setdefault("__unlabeled__", []).append(resp)
                    existing_by_prompt_and_model[pid] = by_model
            print(f"Loaded existing results from {existing_file}")
        except Exception as ex:
            print(f"Warning: Failed to load existing-file '{existing_file}': {ex}")

    # ------------------------------------------------------------------
    # Build tasks
    # ------------------------------------------------------------------
    api_max_tokens = None if (max_tokens is None or max_tokens <= 0) else max_tokens

    collected_by_prompt: Dict[int, Dict] = {
        i: {"prompt_data": p, "responses": []}
        for i, p in enumerate(prompts_data)
    }
    tasks = []
    total_skipped = 0

    for prompt_idx, prompt_data in enumerate(prompts_data):
        pid = prompt_data["task_id"]
        by_model_existing = existing_by_prompt_and_model.get(pid, {})
        for model_name in model_ids:
            existing_entries = by_model_existing.get(model_name, [])
            valid_entries = [
                e for e in existing_entries
                if not (
                    isinstance(e, dict)
                    and str(e.get("response", "")).startswith("[ERROR:")
                )
                and not (
                    isinstance(e, str) and e.startswith("[ERROR:")
                )
            ]
            if valid_entries:
                reuse = valid_entries[:num_responses_per_model]
                for e in reuse:
                    if isinstance(e, dict):
                        collected_by_prompt[prompt_idx]["responses"].append(e)
                    else:
                        collected_by_prompt[prompt_idx]["responses"].append(
                            {"generator": model_name, "response": str(e)}
                        )
                total_skipped += len(reuse)

            have = min(len(valid_entries), num_responses_per_model)
            need = max(0, num_responses_per_model - have)
            for sample_idx in range(have, have + need):
                tasks.append((
                    prompt_idx,
                    model_name,
                    sample_idx,
                    prompt_data["metadata_prompt"],
                ))

    if existing_file:
        print(
            f"Resume mode: reused {total_skipped} existing responses; "
            f"scheduling {len(tasks)} new generations"
        )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    if tasks:
        CHECKPOINT_INTERVAL = 5000
        completed_count = 0
        last_checkpoint_at = 0

        def _build_partial() -> List[Dict]:
            partial: List[Dict] = []
            for idx in range(len(prompts_data)):
                pd = collected_by_prompt[idx]["prompt_data"]
                resps = collected_by_prompt[idx]["responses"]
                if not resps:
                    continue
                partial.append({
                    "id": pd["task_id"],
                    "prompt": pd["metadata_prompt"],
                    "responses": resps,
                })
            return partial

        def _maybe_checkpoint(force: bool = False):
            nonlocal last_checkpoint_at
            if not force and (completed_count - last_checkpoint_at) < CHECKPOINT_INTERVAL:
                return
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                partial = _build_partial()
                with open(output_file + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(partial, f, indent=2, ensure_ascii=False)
                last_checkpoint_at = completed_count
                print(
                    f"Checkpoint: {len(partial)} prompts after "
                    f"{completed_count} new responses"
                )
            except Exception as err:
                print(f"Warning: checkpoint failed: {err}")

        def _generate_single(prompt_text: str, model_name: str) -> tuple:
            conv = format_prompt_as_conversation(prompt_text)
            try:
                text = generate_via_api(
                    model=model_name,
                    messages=conv,
                    max_tokens=api_max_tokens,
                    temperature=temperature,
                    max_retries=api_max_retries,
                    base_url=base_url,
                )
                return model_name, text.strip()
            except Exception as err:
                return model_name, f"[ERROR: {err}]"

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _generate_single, prompt_text, model_name
                ): (prompt_idx, model_name)
                for (prompt_idx, model_name, _, prompt_text) in tasks
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating responses",
            ):
                prompt_idx, model_name = futures[future]
                try:
                    label, text = future.result()
                except Exception as e:
                    label, text = model_name, f"[ERROR: {e}]"
                collected_by_prompt[prompt_idx]["responses"].append(
                    {"generator": label, "response": text}
                )
                completed_count += 1
                if completed_count % CHECKPOINT_INTERVAL == 0:
                    _maybe_checkpoint()

        time.sleep(0.1)  # let threads finish cleanup

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results: List[Dict] = []
    for prompt_idx in range(len(prompts_data)):
        pd = collected_by_prompt[prompt_idx]["prompt_data"]
        resps = collected_by_prompt[prompt_idx]["responses"]
        results.append({
            "id": pd["task_id"],
            "prompt": pd["metadata_prompt"],
            "responses": resps,
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    temp_file = output_file + ".tmp"
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print(
        f"Rollout completed. Saved {len(results)} prompts to {output_file}"
    )
    return True


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompts(
    input_file: str,
    sample_size: Optional[int],
    sequential: bool,
) -> List[Dict]:
    """Load prompts from a JSON file.

    Expected format: a list of dicts, each with a "prompt" key and
    optionally an "id" key.  Example::

        [{"id": "p1", "prompt": "..."}, {"prompt": "..."}]
    """
    if not input_file:
        raise ValueError("Please provide --input-file with a .json file")

    with open(input_file, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

    if sample_size:
        if sequential:
            data = data[:sample_size]
            print(f"Sequential loading: taking first {len(data)} prompts")
        else:
            data = random.sample(data, min(sample_size, len(data)))
            print(f"Random sampling: selected {len(data)} prompts")

    return [
        {
            "task_id": item.get("id", f"prompt_{i}"),
            "metadata_prompt": item["prompt"],
        }
        for i, item in enumerate(data)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_ids = args.models or SOTA_MODELS
    input_file = args.input_file or "data/sample_prompts.json"

    if args.output_filename:
        output_filename = f"{args.output_dir}/{args.output_filename}"
    else:
        if len(model_ids) > 1:
            prefix = "SOTAs" if model_ids == SOTA_MODELS else "multi_model"
            output_filename = (
                f"{args.output_dir}/{prefix}_multi_model_temp_{args.temperature}"
                f"_topp_{args.top_p}_{args.num_prompts}_prompts"
                f"_{args.num_responses_per_model}_responses.json"
            )
        else:
            model_name = model_ids[0].split("/")[-1].replace("/", "_").replace("-", "_")
            base_label = Path(input_file).stem
            output_filename = (
                f"{args.output_dir}/{model_name}_temp_{args.temperature}"
                f"_topp_{args.top_p}_{args.num_prompts}_prompts"
                f"_{args.num_responses_per_model}_responses_{base_label}.json"
            )

    print("--- Configuration ---")
    print(f"Models          : {model_ids}")
    print(f"Input file      : {input_file}")
    print(f"Prompts         : {args.num_prompts}")
    print(f"Responses/model : {args.num_responses_per_model}")
    print(f"Workers         : {args.num_workers}")
    print(f"Max tokens      : {args.max_tokens}")
    print(f"Temperature     : {args.temperature}, Top-p: {args.top_p}")
    print(f"Base URL        : {args.base_url or '(default)'}")
    print(f"Output          : {output_filename}")
    print("-" * 50)

    success = generate_responses(
        input_file=input_file,
        output_file=output_filename,
        num_responses_per_model=args.num_responses_per_model,
        sample_size=args.num_prompts,
        model_ids=model_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sequential=args.sequential,
        api_max_retries=args.api_max_retries,
        num_workers=args.num_workers,
        existing_file=args.existing_file,
        base_url=args.base_url,
    )

    if success:
        print("Response generation completed successfully!")
    else:
        print("Response generation failed!")
        return 1

    time.sleep(0.2)
    return 0


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=ResourceWarning)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
