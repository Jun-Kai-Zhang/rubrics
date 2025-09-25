import json
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Import common utilities
try:
    # Try relative import first (when run as module)
    from .common import (
        format_prompt_as_conversation,
        load_prompts_from_file,
        save_results,
        build_output_filename,
        load_existing_results,
        save_checkpoint,
        cleanup_checkpoint,
        generate_via_api
    )
except ImportError:
    # Fall back to direct import (when run as script)
    from common import (
        format_prompt_as_conversation,
        load_prompts_from_file,
        save_results,
        build_output_filename,
        load_existing_results,
        save_checkpoint,
        cleanup_checkpoint,
        generate_via_api
    )


# Default SOTA models list
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
    # xAI
    # "xai/grok-4-latest",
    # "xai/grok-3-latest",
    # Deepseek
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

def parse_args():
    """
    Parse command line arguments for configuration.
    """
    parser = argparse.ArgumentParser(description="Generate responses from SOTA APIs across multiple providers (API-only)")
    
    # Optional single model override. If not set, we will use the built-in SOTA model list.
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional single model identifier to override the SOTA list."
    )
    
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Number of prompts to process (if omitted, uses full dataset or file)"
    )
    
    parser.add_argument(
        "--num-responses-per-prompt",
        type=int,
        default=1,
        help="Number of responses to generate per model per prompt (default: 4)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Maximum number of tokens to generate (set 0 to not restrict)"
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
        help="Use sequential loading (first N prompts) instead of random sampling for JSON files"
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
        default=64,
        help="Number of parallel workers for API calls (default: 32)"
    )
    
    parser.add_argument(
        "--existing-file",
        type=str,
        default=None,
        help="Path to an existing results JSON. Reuse already-present responses per prompt/model and only generate missing ones."
    )
    
    return parser.parse_args()

def main():
    """
    Main function to load the model, generate responses for the dataset,
    and save them to a JSON file.
    """
    # --- 1. Parse Arguments ---
    args = parse_args()
    
    # --- 2. Configuration ---
    # Allow overriding with a single model if provided
    model_ids = [args.model_id] if args.model_id else SOTA_MODELS
    num_prompts = args.num_prompts
    num_responses_per_prompt = args.num_responses_per_prompt
    
    # API-only; no GPU/vLLM configuration
    
    # Generate output filename
    if args.output_filename:
        output_filename = f"{args.output_dir}/{args.output_filename}"
    else:
        # Consolidated filename for multi-model rollout
        output_filename = f"{args.output_dir}/SOTAs_multi_model_Temperature_{args.temperature}_TopP_{args.top_p}_{num_prompts}_Prompts_{num_responses_per_prompt}_Responses.json"
    
    # Display configuration
    print("--- Configuration ---")
    if args.model_id:
        print(f"Model override: {args.model_id}")
    else:
        print(f"Models: {len(model_ids)} configured (SOTAs list)")
    print(f"Mode: API")
    print(f"Prompts: {num_prompts}, Responses per prompt: {num_responses_per_prompt}")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {output_filename}")
    print("-" * 50)
    

    # API generation
    success = generate_responses(
        input_file=args.input_file or "",
        output_file=output_filename,
        num_responses=num_responses_per_prompt,
        sample_size=num_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sequential=args.sequential,
        api_max_retries=args.api_max_retries,
        num_workers=args.num_workers,
        model_ids=model_ids,
        label_generator=True,
        existing_file=args.existing_file,
    )
    
    if success:
        print("✅ Response generation completed successfully!")
    else:
        print("❌ Response generation failed!")
        return 1
    
    return 0


def generate_responses(
    input_file: str,
    output_file: str,
    num_responses: int,
    sample_size: int,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    sequential: bool = False,
    api_max_retries: int = 3,
    num_workers: int = 64,
    model_ids: Optional[List[str]] = None,
    label_generator: bool = False,
    existing_file: Optional[str] = None,
) -> bool:
    """
    API function to generate responses using vLLM or API-based generation.
    
    Args:
        prompts_file: Path to prompts file (JSON or will use dataset)
        output_file: Path to save generated responses
        num_responses: Number of responses per prompt
        sample_size: Number of prompts to sample
        model_id: Model to use for vLLM generation
        vllm_instance: Pre-initialized vLLM instance (optional)
        tensor_parallel_size: Number of GPUs to use for vLLM
        dataset_name: Name of dataset to use ("OTS", "alpaca_eval", or "ifeval")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        enable_cleanup: If True, cleanup GPU memory after generation (default: True)
        sequential: If True, use sequential loading (first N prompts) for JSON files and datasets
        use_api: If True, use API for generation instead of vLLM
        api_max_retries: Maximum number of retries for API calls
        num_workers: Number of parallel workers for API calls
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load prompts using common function
        prompts_data = load_prompts_from_file(
            input_file=input_file,
            sample_size=sample_size,
            sequential=sequential,
            random_seed=42
        )
        
        actual_num_prompts = len(prompts_data)
        print(f"Loaded {actual_num_prompts} prompts")
        
        results_to_save = []
        
        # Load existing results for resume functionality (multi-model aware)
        _, _, existing_by_prompt_and_model = load_existing_results(
            existing_file=existing_file,
            is_multi_model=True
        )
        
        # API-based generation (multi-model aware)
        if model_ids is None or len(model_ids) == 0:
            raise ValueError("No model_ids provided for API generation")
        print(f"Using API for generation with {len(model_ids)} model(s)")
        print(f"Generating {num_responses} responses per model per prompt...")
        print(f"Using {num_workers} parallel workers")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Interpret non-positive max_tokens as "no cap" for API: do not send the field
        api_max_tokens = None if (max_tokens is None or max_tokens <= 0) else max_tokens

        def generate_single_response(prompt_text: str, model_name: str, response_idx: int) -> tuple[str, str]:
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
        collected_by_prompt: Dict[int, Dict] = {i: {"prompt_data": p_data, "responses": []} for i, p_data in enumerate(prompts_data)}
        
        # Pre-fill with existing responses and enqueue only missing ones per model
        total_skipped = 0
        total_new_tasks = 0
        for prompt_idx, prompt_data in enumerate(prompts_data):
            pid = prompt_data['task_id']
            by_model_existing = existing_by_prompt_and_model.get(pid, {})
            for model_name in model_ids:
                existing_entries = by_model_existing.get(model_name, [])
                if existing_entries:
                    # Keep at most num_responses per model
                    reuse_entries = existing_entries[:num_responses]
                    if label_generator:
                        for e in reuse_entries:
                            if isinstance(e, dict):
                                collected_by_prompt[prompt_idx]["responses"].append(e)
                            else:
                                collected_by_prompt[prompt_idx]["responses"].append({"generator": model_name, "response": str(e)})
                    else:
                        for e in reuse_entries:
                            if isinstance(e, dict):
                                collected_by_prompt[prompt_idx]["responses"].append(e.get("response", ""))
                            else:
                                collected_by_prompt[prompt_idx]["responses"].append(str(e))
                    total_skipped += len(reuse_entries)
                # Determine how many more we need for this model
                have = min(len(existing_entries), num_responses)
                need = max(0, num_responses - have)
                for sample_idx in range(have, have + need):
                    tasks.append((prompt_idx, model_name, sample_idx, prompt_data['metadata_prompt']))
                    total_new_tasks += 1
        if existing_file:
            print(f"Resume mode: reused {total_skipped} existing responses; scheduling {total_new_tasks} new generations")
        
        # Run generation only if there is outstanding work
        if len(tasks) > 0:
            # --- Checkpointing config ---
            # Save a checkpoint every N newly generated responses (not counting reused)
            ROLLOUTS_CHECKPOINT_INTERVAL = 5000
            completed_rollouts = 0
            last_checkpoint_at = 0

            def build_partial_results_for_checkpoint() -> List[Dict]:
                """
                Build a partial results list suitable for saving checkpoints.
                Includes any prompts for which we have at least one response
                (reused or newly generated), preserving original order.
                """
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
                partial_results = build_partial_results_for_checkpoint()
                if save_checkpoint(output_file, partial_results):
                    last_checkpoint_at = completed_rollouts
                    print(f"Checkpoint: saved {len(partial_results)} prompts after {completed_rollouts} new responses")

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
                    entry = {"generator": model_label, "response": resp_text} if label_generator else resp_text
                    collected_by_prompt[prompt_idx]["responses"].append(entry)
                    # Count newly generated response and maybe checkpoint
                    completed_rollouts += 1
                    if completed_rollouts % ROLLOUTS_CHECKPOINT_INTERVAL == 0:
                        maybe_save_checkpoint()
        
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
        
        # Save intermediate results snapshot
        if len(results_to_save) > 0:
            save_checkpoint(output_file, results_to_save)
        
        print(f"Generated responses for {len(results_to_save)} prompts across {len(model_ids)} model(s)")
        
        # Save results
        save_results(results_to_save, output_file)
        
        # Clean up checkpoint file
        cleanup_checkpoint(output_file)
        
        print(f"Rollout completed. Saved {len(results_to_save)} prompts to {output_file}")
            
        return True
        
    except Exception as e:
        print(f"Error in generate_responses_api: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
