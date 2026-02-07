import json
import argparse
import atexit
import sys
from pathlib import Path

# Conditional import for vLLM (only needed when using vLLM backend)
try:
    from vllm import LLM, SamplingParams
    import torch
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
    torch = None

# ---------------------------------------------------------------------------
# Local imports – prefer package-relative. Work under both module and script run.
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent))
from generator.utils import (
    generate_via_api,
)
from typing import Optional, List, Dict
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def format_prompt_as_conversation(prompt_text):
    """
    Converts a prompt text into a conversation format for API calls.
    """
    return [{"role": "user", "content": prompt_text}]

def parse_args():
    """
    Parse command line arguments for configuration.
    """
    parser = argparse.ArgumentParser(description="Generate responses using API")

    # Accept list of models
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        default=None,
        help="Model identifier(s) for API. Can specify multiple models."
    )

    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to process"
    )

    parser.add_argument(
        "--num-responses-per-model",
        type=int,
        default=1,
        help="Number of responses to generate per model per prompt (default: 1)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Maximum number of tokens to generate (set 0 to use provider default)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for results (default: data)"
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Custom output filename (if not provided, will be auto-generated)"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to an input file containing prompts: CSV (expects columns task_id, metadata_prompt) or JSON"
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential loading (first N prompts) instead of random sampling for files"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "api"],
        default="api",
        help="Backend to use for generation: 'vllm' or 'api' (default: api)"
    )

    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls (default: 3)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for API calls (default: 16)"
    )

    parser.add_argument(
        "--existing-file",
        type=str,
        default=None,
        help="Path to an existing results JSON. Reuse existing responses per prompt and only generate missing ones."
    )

    return parser.parse_args()

def main():
    """
    Main function to load the model, generate responses for the dataset,
    and save them to a JSON file.
    """
    # --- 1. Parse Arguments ---
    args = parse_args()

    backend = args.backend

    # Default SOTA model list for API mode
    SOTA_MODELS = [
        # OpenAI
        "openai/gpt-5",
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4.1",
        "openai/o3",
        "openai/o1-2024-12-17",
        "openai/o4-mini",
        # Google
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
        # Anthropic
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3-7-sonnet-20250219",
        "fireworks_ai/deepseek-v3",
        "fireworks_ai/deepseek-r1",
        # Kimi
        "fireworks_ai/kimi-k2-instruct",
        # GLM
        "fireworks_ai/glm-4p5",
        # Qwen
        "fireworks_ai/qwen3-235b-a22b-instruct-2507",
        # Mistral
        "mistral/mistral-medium-latest"
    ]

    # Resolve model(s) based on backend
    if backend == "api":
        if args.models:
            model_ids = args.models
        else:
            # Default to SOTA models for API
            model_ids = SOTA_MODELS
    else:  # vllm backend
        if args.models and len(args.models) > 0:
            # For vLLM, only use the first model (vLLM doesn't support multi-model)
            model_ids = [args.models[0]]
        else:
            # Default vLLM model
            model_ids = ["checkpoints/Qwen2.5-7B-sft/Qwen2.5-7B-sft-alpaca_and_dolly/global_step_259"]

    num_prompts = args.num_prompts
    num_responses_per_model = args.num_responses_per_model
    input_file = args.input_file or "data/ots_train_1k.csv"

    # Calculate tensor parallel size for vLLM
    tensor_parallel_size = 1
    if backend == "vllm" and torch and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        tensor_parallel_size = max(1, num_gpus)
        print(f"Detected {num_gpus} GPU(s), setting tensor_parallel_size to {tensor_parallel_size}")

    # Generate output filename
    if args.output_filename:
        output_filename = f"{args.output_dir}/{args.output_filename}"
    else:
        if backend == "api" and len(model_ids) > 1:
            # Multi-model API output
            prefix = "SOTAs" if model_ids == SOTA_MODELS else "multi_model"
            output_filename = f"{args.output_dir}/{prefix}_multi_model_temp_{args.temperature}_topp_{args.top_p}_{num_prompts}_prompts_{num_responses_per_model}_responses.json"
        else:
            # Single model (API or vLLM)
            model_name = model_ids[0].split("/")[-1].replace("/", "_").replace("-", "_")
            base_label = Path(input_file).stem if input_file else "prompts"
            output_filename = f"{args.output_dir}/{model_name}_temp_{args.temperature}_topp_{args.top_p}_{num_prompts}_prompts_{num_responses_per_model}_responses_{base_label}.json"

    # Display configuration
    print("--- Configuration ---")
    print(f"Backend: {backend.upper()}")
    print(f"Models: {model_ids}")
    print(f"Input file: {input_file}")
    print(f"Prompts: {num_prompts}, Responses per model per prompt: {num_responses_per_model}")
    if backend == "api":
        print(f"Parallel workers: {args.num_workers}")
    else:
        print(f"GPUs: {tensor_parallel_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {output_filename}")
    print("-" * 50)


    success = generate_responses(
        input_file=input_file,
        output_file=output_filename,
        num_responses_per_model=num_responses_per_model,
        sample_size=num_prompts,
        model_ids=model_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sequential=args.sequential,
        backend=backend,
        tensor_parallel_size=tensor_parallel_size,
        api_max_retries=args.api_max_retries,
        num_workers=args.num_workers,
        existing_file=args.existing_file
    )

    if success:
        print("✅ Response generation completed successfully!")
    else:
        print("❌ Response generation failed!")
        return 1

    # Allow background threads to complete cleanup before exit
    import time
    time.sleep(0.2)

    return 0


def generate_responses(
    input_file: str,
    output_file: str,
    num_responses_per_model: int,
    sample_size: int,
    model_ids: List[str],
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    sequential: bool = False,
    backend: str = "api",
    tensor_parallel_size: int = 1,
    vllm_instance: Optional[LLM] = None,
    enable_cleanup: bool = True,
    api_max_retries: int = 3,
    num_workers: int = 64,
    existing_file: Optional[str] = None
) -> bool:
    """
    API function to generate responses using API or vLLM backend.

    Args:
        input_file: Path to prompts file (.csv with task_id, metadata_prompt) or JSON
        output_file: Path to save generated responses
        num_responses_per_model: Number of responses per model per prompt
        sample_size: Number of prompts to sample
        model_ids: List of models to use for generation
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        sequential: If True, use sequential loading (first N prompts) for files
        backend: Backend to use: "api" or "vllm"
        tensor_parallel_size: Number of GPUs to use for vLLM
        vllm_instance: Pre-initialized vLLM instance (optional)
        enable_cleanup: If True, cleanup GPU memory after generation (default: True)
        api_max_retries: Maximum number of retries for API calls
        num_workers: Number of parallel workers for API calls
        existing_file: Path to existing results for resume support

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Set random seed for consistent sampling
        import random
        random.seed(42)

        # Load prompts
        if input_file and input_file.endswith('.json'):
            # Load from JSON file
            with open(input_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            # New: Detect initial_rubrics.json format: top-level object with "rubrics" array
            if isinstance(data, dict) and isinstance(data.get('rubrics'), list):
                rubrics_list = data.get('rubrics', [])
                print(f"Detected initial_rubrics format with {len(rubrics_list)} total rubrics")
                if sample_size:
                    if sequential:
                        rubrics_list = rubrics_list[:sample_size]
                        print(f"Using sequential loading: taking first {len(rubrics_list)} rubrics")
                    else:
                        import random
                        rubrics_list = random.sample(rubrics_list, min(sample_size, len(rubrics_list)))
                        print(f"Using random sampling: selected {len(rubrics_list)} rubrics")
                prompts_data = [
                    {
                        'task_id': (item.get('id') or f"rubric_{idx}"),
                        'metadata_prompt': (item.get('prompt') or ''),
                    }
                    for idx, item in enumerate(rubrics_list)
                ]
            # Check if this is wildchat_filtering format
            elif isinstance(data, list) and data and isinstance(data[0], dict) and 'conversation_id' in data[0] and 'instruction' in data[0]:
                # Handle wildchat_filtering/final_prompts.json format
                print(f"Detected wildchat_filtering format with {len(data)} total prompts")
                if sample_size:
                    data = data[:sample_size]
                    print(f"Using sequential loading: taking first {len(data)} prompts")
                prompts_data = [{'task_id': item['conversation_id'], 'metadata_prompt': item['instruction']} for item in data]
            # PPE CSV-derived JSON or similar
            elif isinstance(data, list) and data and isinstance(data[0], dict) and ('prompts' in data[0] or 'prompt' in data[0]):
                print(f"Detected PPE/JSON prompts format with {len(data)} total prompts")
                if sample_size:
                    if sequential:
                        data = data[:sample_size]
                        print(f"Using sequential loading: taking first {len(data)} prompts")
                    else:
                        import random
                        data = random.sample(data, min(sample_size, len(data)))
                        print(f"Using random sampling: selected {len(data)} prompts")
                prompts_data = [
                    {
                        'task_id': (item.get('ids') or item.get('id') or f"json_{idx}"),
                        'metadata_prompt': (item.get('prompts') or item.get('prompt') or ''),
                    }
                    for idx, item in enumerate(data)
                ]
            elif isinstance(data, list):
                # Check if this is health data format (with criteria fields)
                if data and isinstance(data[0], dict) and 'criteria_1' in data[0]:
                    print(f"Detected health data format with {len(data)} total prompts")
                    if sample_size:
                        if sequential:
                            data = data[:sample_size]
                            print(f"Using sequential loading: taking first {len(data)} prompts")
                        else:
                            import random
                            data = random.sample(data, min(sample_size, len(data)))
                            print(f"Using random sampling: selected {len(data)} prompts")
                    prompts_data = [
                        {
                            'task_id': item.get('id', f"health_{idx}"),
                            'metadata_prompt': item.get('prompt', ''),
                            # Preserve additional health data fields
                            'health_data': {k: v for k, v in item.items() if k not in ['id', 'prompt']}
                        }
                        for idx, item in enumerate(data)
                    ]
                # Handle original format (list of {id, prompt})
                else:
                    print(f"Detected standard format with {len(data)} total prompts")
                    if sample_size:
                        if sequential:
                            data = data[:sample_size]
                            print(f"Using sequential loading: taking first {len(data)} prompts")
                        else:
                            import random
                            data = random.sample(data, min(sample_size, len(data)))
                            print(f"Using random sampling: selected {len(data)} prompts")
                    prompts_data = [{'task_id': item.get('id', f"json_{idx}"), 'metadata_prompt': item.get('prompt', '')} for idx, item in enumerate(data)]
            else:
                raise ValueError("Unsupported JSON format for prompts file")
        elif input_file and input_file.endswith('.csv'):
            # Load from CSV (expects task_id, metadata_prompt)
            import pandas as pd
            print(f"Loading prompts from CSV: {input_file}")
            df = pd.read_csv(input_file, low_memory=False)
            required_cols = {"task_id", "metadata_prompt"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"CSV is missing required columns: {missing}")
            if sample_size is not None:
                if sequential:
                    df = df.head(min(sample_size, len(df)))
                    print(f"Using sequential loading: taking first {len(df)} prompts")
                else:
                    df = df.sample(min(sample_size, len(df)), random_state=42)
                    print(f"Using random sampling: selected {len(df)} prompts")
            prompts_data = df[["task_id", "metadata_prompt"]].to_dict('records')
        else:
            raise ValueError("Please provide --input-file with .csv or .json")

        actual_num_prompts = len(prompts_data)
        print(f"Loaded {actual_num_prompts} prompts")

        results_to_save = []

        # Optional: Load existing results to resume/skip work
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
                        responses = entry.get("responses", [])
                        by_model: Dict[str, List] = {}
                        for resp in responses:
                            if isinstance(resp, dict) and "generator" in resp:
                                model_name = resp.get("generator")
                                by_model.setdefault(model_name, []).append(resp)
                            else:
                                # Unlabeled response
                                by_model.setdefault("__unlabeled__", []).append(resp)
                        existing_by_prompt_and_model[pid] = by_model
                else:
                    print("Warning: existing-file is not a list; ignoring for resume.")
                print(f"Loaded existing results from {existing_file}")
            except Exception as ex:
                print(f"Warning: Failed to load existing-file '{existing_file}': {ex}")

        # Choose backend: API or vLLM
        if backend == "api":
            # API-based generation (multi-model aware)
            print(f"Using API for generation with {len(model_ids)} model(s)")
            print(f"Generating {num_responses_per_model} responses per model per prompt...")
            print(f"Using {num_workers} parallel workers")

            # Interpret non-positive max_tokens as "no cap" for API
            api_max_tokens = None if (max_tokens is None or max_tokens <= 0) else max_tokens

            def generate_single_response(prompt_text: str, model_name: str, response_idx: int) -> tuple:
                """Return (model_name, response_text) for a single sample."""
                conversation = format_prompt_as_conversation(prompt_text)
                try:
                    response_text = generate_via_api(
                        model=model_name,
                        messages=conversation,
                        max_tokens=api_max_tokens,
                        temperature=temperature,
                        max_retries=api_max_retries,
                    )
                    return model_name, response_text.strip()
                except Exception as api_error:
                    return model_name, f"[ERROR: {str(api_error)}]"

            # Prepare tasks: (prompt_idx, model_name, sample_idx)
            tasks = []
            # Container for collected results per prompt
            collected_by_prompt: Dict[int, Dict] = {
                i: {"prompt_data": p_data, "responses": []}
                for i, p_data in enumerate(prompts_data)
            }

            # Pre-fill with existing responses and enqueue only missing ones per model
            total_skipped = 0
            total_new_tasks = 0
            for prompt_idx, prompt_data in enumerate(prompts_data):
                pid = prompt_data['task_id']
                by_model_existing = existing_by_prompt_and_model.get(pid, {})
                for model_name in model_ids:
                    existing_entries = by_model_existing.get(model_name, [])
                    # Filter out error responses
                    valid_entries = []
                    for e in existing_entries:
                        response_text = ""
                        if isinstance(e, dict):
                            response_text = e.get("response", "")
                        else:
                            response_text = str(e)
                        if not response_text.startswith("[ERROR:"):
                            valid_entries.append(e)

                    if valid_entries:
                        # Keep at most num_responses_per_model per model
                        reuse_entries = valid_entries[:num_responses_per_model]
                        for e in reuse_entries:
                            if isinstance(e, dict):
                                collected_by_prompt[prompt_idx]["responses"].append(e)
                            else:
                                collected_by_prompt[prompt_idx]["responses"].append({
                                    "generator": model_name,
                                    "response": str(e)
                                })
                        total_skipped += len(reuse_entries)

                    # Determine how many more we need for this model
                    have = min(len(valid_entries), num_responses_per_model)
                    need = max(0, num_responses_per_model - have)
                    for sample_idx in range(have, have + need):
                        tasks.append((prompt_idx, model_name, sample_idx, prompt_data['metadata_prompt']))
                        total_new_tasks += 1

            if existing_file:
                print(f"Resume mode: reused {total_skipped} existing responses; scheduling {total_new_tasks} new generations")

            # Run generation if needed
            if len(tasks) > 0:
                # Checkpointing config
                ROLLOUTS_CHECKPOINT_INTERVAL = 5000
                completed_rollouts = 0
                last_checkpoint_at = 0

                def build_partial_results_for_checkpoint() -> List[Dict]:
                    """Build partial results list for checkpoints."""
                    partial_results: List[Dict] = []
                    for idx in range(len(prompts_data)):
                        prompt_data = collected_by_prompt[idx]["prompt_data"]
                        responses = collected_by_prompt[idx]["responses"]
                        if not responses:
                            continue
                        partial_results.append({
                            "id": prompt_data['task_id'],
                            "prompt": prompt_data['metadata_prompt'],
                            "responses": responses,
                        })
                    return partial_results

                def maybe_save_checkpoint(force: bool = False):
                    nonlocal last_checkpoint_at
                    if not force and (completed_rollouts - last_checkpoint_at) < ROLLOUTS_CHECKPOINT_INTERVAL:
                        return
                    try:
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        partial_results = build_partial_results_for_checkpoint()
                        with open(output_file + ".tmp", "w", encoding="utf-8") as f:
                            json.dump(partial_results, f, indent=2, ensure_ascii=False)
                        last_checkpoint_at = completed_rollouts
                        print(f"Checkpoint: saved {len(partial_results)} prompts after {completed_rollouts} new responses → {output_file}.tmp")
                    except Exception as ckpt_err:
                        print(f"Warning: failed to save checkpoint: {ckpt_err}")

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_task = {
                        executor.submit(generate_single_response, prompt_text, model_name, sample_idx): (prompt_idx, model_name, sample_idx)
                        for (prompt_idx, model_name, sample_idx, prompt_text) in tasks
                    }
                    for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Generating responses"):
                        prompt_idx, model_name, sample_idx = future_to_task[future]
                        try:
                            model_label, resp_text = future.result()
                        except Exception as e:
                            model_label, resp_text = model_name, f"[ERROR: {str(e)}]"
                        entry = {"generator": model_label, "response": resp_text}
                        collected_by_prompt[prompt_idx]["responses"].append(entry)
                        # Count and checkpoint
                        completed_rollouts += 1
                        if completed_rollouts % ROLLOUTS_CHECKPOINT_INTERVAL == 0:
                            maybe_save_checkpoint()

                # Allow threads to complete cleanup
                import time
                time.sleep(0.1)

            # Build results preserving original prompt order
            for prompt_idx in range(len(prompts_data)):
                prompt_data = collected_by_prompt[prompt_idx]["prompt_data"]
                responses = collected_by_prompt[prompt_idx]["responses"]
                result_entry = {
                    "id": prompt_data['task_id'],
                    "prompt": prompt_data['metadata_prompt'],
                    "responses": responses,
                }
                # Include health data if present
                if 'health_data' in prompt_data:
                    result_entry.update(prompt_data['health_data'])
                results_to_save.append(result_entry)

            print(f"Generated responses for {len(results_to_save)} prompts across {len(model_ids)} model(s)")

        else:  # vLLM backend
            # vLLM-based generation
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Please install vLLM or use --backend api for API-based generation.")

            # Initialize vLLM if not provided
            if vllm_instance is None:
                print(f"Initializing vLLM with {model_ids[0]}...")
                llm = LLM(
                    model=model_ids[0],
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype='bfloat16',
                )
            else:
                llm = vllm_instance

            # For vLLM, we treat it as single-model with multiple responses
            # Convert existing data format (API multi-model) to simple format for vLLM
            existing_by_id: Dict[str, List[str]] = {}
            existing_by_prompt_text: Dict[str, List[str]] = {}
            if existing_file and os.path.exists(existing_file):
                try:
                    with open(existing_file, "r", encoding="utf-8") as efp:
                        existing_data = json.load(efp)
                    if isinstance(existing_data, list):
                        for entry in existing_data:
                            if not isinstance(entry, dict):
                                continue
                            pid = entry.get("id") or entry.get("task_id")
                            ptxt = entry.get("prompt")
                            raw_resps = entry.get("responses", [])
                            coerced: List[str] = []
                            if isinstance(raw_resps, list):
                                for r in raw_resps:
                                    if isinstance(r, dict):
                                        val = r.get("response")
                                        if isinstance(val, str):
                                            coerced.append(val)
                                    elif isinstance(r, str):
                                        coerced.append(r)
                            if pid is not None:
                                existing_by_id[str(pid)] = coerced
                            if isinstance(ptxt, str) and ptxt.strip():
                                existing_by_prompt_text[ptxt] = coerced
                    print(f"Loaded existing results from {existing_file}")
                except Exception as ex:
                    print(f"Warning: Failed to load existing-file '{existing_file}': {ex}")

            # Resume-aware vLLM generation
            from collections import defaultdict
            reused_by_id: Dict[str, List[str]] = {}
            need_by_id: Dict[str, int] = {}
            id_to_prompt: Dict[str, Dict] = {}
            for prompt_data in prompts_data:
                pid = str(prompt_data['task_id'])
                id_to_prompt[pid] = prompt_data
                ptxt = prompt_data['metadata_prompt']
                pre_existing = existing_by_id.get(pid) or existing_by_prompt_text.get(ptxt, [])
                reuse_count = min(len(pre_existing), num_responses_per_model)
                reused_by_id[pid] = pre_existing[:reuse_count]
                need_by_id[pid] = max(0, num_responses_per_model - reuse_count)
            total_skipped = sum(len(v) for v in reused_by_id.values())
            total_needed = sum(need_by_id.values())
            if existing_file:
                print(f"Resume mode: reused {total_skipped} existing responses; need {total_needed} new generations")

            new_by_id: Dict[str, List[str]] = {pid: [] for pid in need_by_id.keys()}
            prompts_by_need: Dict[int, List[str]] = defaultdict(list)
            for pid, need_cnt in need_by_id.items():
                if need_cnt > 0:
                    prompts_by_need[need_cnt].append(pid)

            # Interpret non-positive max_tokens as very large number for vLLM
            vllm_max_tokens = max_tokens if (max_tokens is not None and max_tokens > 0) else 1_000_000

            for need_cnt, pid_list in sorted(prompts_by_need.items()):
                print(f"vLLM: generating {need_cnt} responses for {len(pid_list)} prompt(s)...")
                conversations = [format_prompt_as_conversation(id_to_prompt[pid]['metadata_prompt']) for pid in pid_list]
                sampling_params = SamplingParams(
                    n=need_cnt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=vllm_max_tokens,
                )
                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling_params,
                    use_tqdm=True
                )
                for idx, output in enumerate(outputs):
                    pid = pid_list[idx]
                    gen_texts = [out.text.strip() for out in output.outputs]
                    new_by_id[pid].extend(gen_texts)

            # Build final results preserving original order
            for prompt_data in prompts_data:
                pid = str(prompt_data['task_id'])
                reused = reused_by_id.get(pid, [])
                newly = new_by_id.get(pid, [])
                combined = (reused + newly)[:num_responses_per_model]
                while len(combined) < num_responses_per_model:
                    combined.append("[ERROR: Response not generated]")

                # Format responses with generator label for consistency
                formatted_responses = []
                for resp_text in combined:
                    formatted_responses.append({
                        "generator": model_ids[0],
                        "response": resp_text
                    })

                result_entry = {
                    "id": prompt_data['task_id'],
                    "prompt": prompt_data['metadata_prompt'],
                    "responses": formatted_responses
                }
                # Include health data if present
                if 'health_data' in prompt_data:
                    result_entry.update(prompt_data['health_data'])
                results_to_save.append(result_entry)

            print(f"Generated responses for {len(results_to_save)} prompts using vLLM")

            # Clean up vLLM instance if we created it
            if enable_cleanup and vllm_instance is None and 'llm' in locals():
                print("Cleaning up vLLM instance...")
                try:
                    # Try to properly shutdown the LLM if it has a shutdown method
                    if hasattr(llm, 'shutdown'):
                        llm.shutdown()
                    elif hasattr(llm, 'stop'):
                        llm.stop()
                except Exception as e:
                    print(f"Warning: Error during LLM shutdown: {e}")

                # Delete the instance
                del llm

                # Force garbage collection and clear GPU cache
                import gc
                gc.collect()

                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Kill any remaining vLLM worker processes
                import subprocess
                try:
                    # Find and kill any VllmWorkerProcess
                    result = subprocess.run(['pkill', '-f', 'VllmWorkerProcess'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Killed remaining vLLM worker processes")
                except Exception as e:
                    print(f"Warning: Could not kill vLLM processes: {e}")

                # Additional cleanup for distributed processes
                try:
                    # Reset NCCL if it was initialized
                    if hasattr(torch.distributed, 'is_initialized') and torch.distributed.is_initialized():
                        torch.distributed.destroy_process_group()
                        print("Destroyed torch distributed process group")
                except Exception as e:
                    print(f"Warning: Could not destroy process group: {e}")

                print("Cleanup completed.")

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        # Clean up temporary file if it exists
        temp_file = output_file + ".tmp"
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print(f"Rollout completed. Saved {len(results_to_save)} prompts to {output_file}")

        return True

    except Exception as e:
        print(f"Error in generate_responses: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    import warnings

    # Suppress multiprocess resource tracker warnings during shutdown
    warnings.filterwarnings("ignore", category=ResourceWarning)

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
