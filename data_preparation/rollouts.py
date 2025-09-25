import json
import argparse
import torch
import sys
from pathlib import Path
from typing import Optional, List, Dict
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import random
import gc
import subprocess

# Conditional import for vLLM (only needed when not using API mode)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

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



def parse_args():
    """
    Parse command line arguments for configuration.
    """
    parser = argparse.ArgumentParser(description="Generate responses using vLLM or API for reward bench dataset")
    
    # Unified model-id for both backends (replaces separate api-model)
    parser.add_argument(
        "--model-id", 
        type=str, 
        default=None,
        help="Model identifier. For vLLM, pass a local/HF model id or path; for API, pass the API model name."
    )
    
    parser.add_argument(
        "--num-prompts", 
        type=int, 
        default=None,
        help="Number of prompts to process"
    )
    
    parser.add_argument(
        "--num-responses-per-prompt", 
        type=int, 
        default=16,
        help="Number of responses to generate per prompt (default: 64)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=0,
        help="Maximum number of tokens to generate (set 0 to use provider default / effectively unlimited for API)"
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
    
    # Removed: --test and dataset selection; we now load from CSV/JSON files
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
    # Backend selection replaces --use-api
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "api"],
        default="api",
        help="Backend to use for generation: 'vllm' or 'api'"
    )
    # Removed legacy --api-model (use --model-id with --backend api)
    
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls (default: 3)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=128,
        help="Number of parallel workers for API calls (default: 64)"
    )
    
    parser.add_argument(
        "--existing-file",
        type=str,
        default=None,
        help="Path to an existing results JSON. Reuse existing responses per prompt and only generate missing ones to reach --num-responses-per-prompt."
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
    # Determine backend and model from unified flags
    backend = args.backend
    # Resolve model precedence: --model-id (primary) or choose sensible defaults per backend
    if args.model_id:
        model_id = args.model_id
    else:
        model_id = (
            "gemini/gemini-2.5-flash-lite" if backend == "api" else
            "checkpoints/Qwen2.5-7B-sft/Qwen2.5-7B-sft-alpaca_and_dolly/global_step_259"
        )
    num_prompts = args.num_prompts
    num_responses_per_prompt = args.num_responses_per_prompt
    # Determine input file; default to OTS CSV if not provided
    input_file = args.input_file or "data/ots_train_1k.csv"
    
    # Calculate tensor parallel size based on available GPUs
    num_gpus = torch.cuda.device_count()
    tensor_parallel_size = max(1, num_gpus)  # At least 1, use all available GPUs
    
    print(f"Detected {num_gpus} GPU(s), setting tensor_parallel_size to {tensor_parallel_size}")
    
    # Generate output filename
    if args.output_filename:
        output_filename = f"{args.output_dir}/{args.output_filename}"
    else:
        model_name = model_id.split("/")[-1].replace("/", "_").replace("-", "_")
        base_label = Path(input_file).stem if input_file else "prompts"
        output_filename = f"{args.output_dir}/Policy_Model_{model_name}_Temperature_{args.temperature}_TopP_{args.top_p}_{num_prompts}_Prompts_{num_responses_per_prompt}_Responses_{base_label}.json"
    
    # Display configuration
    print("--- Configuration ---")
    print(f"Model: {model_id}")
    print(f"Mode: {'API' if backend == 'api' else 'vLLM'}")
    print(f"Input file: {input_file}")
    print(f"Prompts: {num_prompts}, Responses per prompt: {num_responses_per_prompt}")
    if backend != 'api':
        print(f"GPUs: {num_gpus} (tensor parallel size: {tensor_parallel_size})")
    else:
        print(f"Parallel workers: {args.num_workers}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {output_filename}")
    print("-" * 50)
    

    success = generate_responses(
        input_file=input_file,
        output_file=output_filename,
        num_responses=num_responses_per_prompt,
        sample_size=num_prompts,
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sequential=args.sequential,
        use_api=(backend == 'api'),
        api_max_retries=args.api_max_retries,
        num_workers=args.num_workers,
        existing_file=args.existing_file
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
    model_id: str,
    vllm_instance: Optional[LLM] = None,
    tensor_parallel_size: Optional[int] = None,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    enable_cleanup: bool = True,
    sequential: bool = False,
    use_api: bool = False,
    api_max_retries: int = 3,
    num_workers: int = 64,
    existing_file: Optional[str] = None
) -> bool:
    """
    API function to generate responses using vLLM or API-based generation.
    
    Args:
        input_file: Path to prompts file (.csv with task_id, metadata_prompt) or JSON
        output_file: Path to save generated responses
        num_responses: Number of responses per prompt
        sample_size: Number of prompts to sample
        model_id: Model to use for vLLM generation
        vllm_instance: Pre-initialized vLLM instance (optional)
        tensor_parallel_size: Number of GPUs to use for vLLM
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        enable_cleanup: If True, cleanup GPU memory after generation (default: True)
        sequential: If True, use sequential loading (first N prompts) for files
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
        
        # Load existing results for resume functionality
        existing_by_id, existing_by_prompt_text, _ = load_existing_results(
            existing_file=existing_file,
            is_multi_model=False
        )
        
        if use_api:
            # API-based generation
            print(f"Using API for generation with model: {model_id}")
            print(f"Generating {num_responses} responses per prompt...")
            
            # Process prompts with progress bar
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading

            # Interpret non-positive max_tokens as "no cap" for API: do not send the field
            api_max_tokens = None if (max_tokens is None or max_tokens <= 0) else max_tokens
            
            # --- Checkpointing config ---
            # We checkpoint after a certain number of completed rollouts (individual generations)
            ROLLOUTS_CHECKPOINT_INTERVAL = 5000
            completed_rollouts = 0
            last_checkpoint_at = 0

            def build_partial_results_for_checkpoint() -> List[Dict]:
                """
                Build a partial results list suitable for saving checkpoints.
                Only include responses that form a contiguous prefix from index 0 so
                resume logic can safely reuse them.
                """
                partial_results: List[Dict] = []
                for prompt_data in prompts_data:
                    task_id = prompt_data['task_id']
                    if task_id not in all_responses:
                        continue
                    resp_map = all_responses[task_id]['responses']
                    # gather contiguous responses starting from index 0
                    contiguous: List[str] = []
                    idx = 0
                    while idx in resp_map:
                        contiguous.append(resp_map[idx])
                        idx += 1
                        if len(contiguous) >= num_responses:
                            break
                    if len(contiguous) == 0:
                        continue
                    partial_results.append({
                        "id": task_id,
                        "prompt": prompt_data['metadata_prompt'],
                        "responses": contiguous
                    })
                return partial_results

            def maybe_save_checkpoint(force: bool = False):
                """Save a checkpoint to output_file.tmp if interval reached or forced."""
                nonlocal last_checkpoint_at
                if not force and (completed_rollouts - last_checkpoint_at) < ROLLOUTS_CHECKPOINT_INTERVAL:
                    return
                partial_results = build_partial_results_for_checkpoint()
                if save_checkpoint(output_file, partial_results):
                    last_checkpoint_at = completed_rollouts
                    print(f"Checkpoint saved with {len(partial_results)} prompts after {completed_rollouts} rollouts")
            
            def generate_single_response(prompt_data, response_idx):
                """Generate a single response for a prompt using the API."""
                conversation = format_prompt_as_conversation(prompt_data['metadata_prompt'])
                
                # Call the API - let generate_via_api handle its own retries
                try:
                    response = generate_via_api(
                        model=model_id,
                        messages=conversation,
                        max_tokens=api_max_tokens,
                        temperature=temperature,
                        max_retries=api_max_retries
                    )
                    return response.strip()
                except Exception as api_error:
                    print(f"API error for {prompt_data['task_id']}, response {response_idx}: {api_error}")
                    return f"[ERROR: {str(api_error)}]"
            
            # Process prompts with parallel API calls (similar to generate_rubrics_without_responses.py)
            print(f"Generating {num_responses} responses for {actual_num_prompts} prompts using API...")
            print(f"Using {num_workers} parallel workers")
            
            # Prepare all tasks with resume support
            all_tasks = []
            all_responses = {}  # Store responses by (task_id, resp_idx)
            total_skipped = 0
            total_new_tasks = 0
            for prompt_data in prompts_data:
                task_id = prompt_data['task_id']
                prompt_text = prompt_data['metadata_prompt']
                # Determine existing responses for this prompt (prefer id, fallback to prompt text)
                pre_existing = existing_by_id.get(str(task_id)) or existing_by_prompt_text.get(prompt_text, [])
                reuse_count = min(len(pre_existing), num_responses)
                if reuse_count > 0:
                    # Seed existing responses into index slots [0..reuse_count-1]
                    for i in range(reuse_count):
                        if task_id not in all_responses:
                            all_responses[task_id] = {'prompt_data': prompt_data, 'responses': {}}
                        all_responses[task_id]['responses'][i] = pre_existing[i]
                    total_skipped += reuse_count
                # Enqueue only missing ones
                for resp_idx in range(reuse_count, num_responses):
                    all_tasks.append((prompt_data, resp_idx))
                    total_new_tasks += 1
            if existing_file:
                print(f"Resume mode: reused {total_skipped} existing responses; scheduling {total_new_tasks} new generations")
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(generate_single_response, task[0], task[1]): task for task in all_tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                    task = futures[future]
                    prompt_data, resp_idx = task
                    
                    try:
                        response = future.result()
                        task_id = prompt_data['task_id']
                        
                        if task_id not in all_responses:
                            all_responses[task_id] = {'prompt_data': prompt_data, 'responses': {}}
                        all_responses[task_id]['responses'][resp_idx] = response
                        
                    except Exception as e:
                        print(f"Worker failed for {prompt_data['task_id']}, response {resp_idx}: {e}")
                        # Store error response
                        task_id = prompt_data['task_id']
                        if task_id not in all_responses:
                            all_responses[task_id] = {'prompt_data': prompt_data, 'responses': {}}
                        all_responses[task_id]['responses'][resp_idx] = f"[ERROR: {str(e)}]"

                    # Count completed rollout and maybe checkpoint
                    completed_rollouts += 1
                    if completed_rollouts % ROLLOUTS_CHECKPOINT_INTERVAL == 0:
                        maybe_save_checkpoint()
            
            # Build results in the original order
            for prompt_data in prompts_data:
                task_id = prompt_data['task_id']
                if task_id in all_responses:
                    responses = []
                    for i in range(num_responses):
                        if i in all_responses[task_id]['responses']:
                            responses.append(all_responses[task_id]['responses'][i])
                        else:
                            responses.append("[ERROR: Response not generated]")
                    
                    result_entry = {
                        "id": task_id,
                        "prompt": prompt_data['metadata_prompt'],
                        "responses": responses
                    }
                    # Include health data if present
                    if 'health_data' in prompt_data:
                        result_entry.update(prompt_data['health_data'])
                    results_to_save.append(result_entry)
                
                # Save intermediate results periodically
                if len(results_to_save) % 10 == 0 and len(results_to_save) > 0:
                    save_checkpoint(output_file, results_to_save)
            
            print(f"Generated responses for {len(results_to_save)} prompts")
        
        else:
            # vLLM-based generation
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM is not installed. Please install vLLM or run with --backend api for API-based generation.")
            
            # Initialize vLLM if not provided
            if vllm_instance is None:
                if tensor_parallel_size is None:
                    tensor_parallel_size = max(1, torch.cuda.device_count())
                
                print(f"Initializing vLLM with {model_id}...")
                llm = LLM(
                    model=model_id,
                    trust_remote_code=True,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype='bfloat16',
                )
            else:
                llm = vllm_instance
            
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
                reuse_count = min(len(pre_existing), num_responses)
                reused_by_id[pid] = pre_existing[:reuse_count]
                need_by_id[pid] = max(0, num_responses - reuse_count)
            total_skipped = sum(len(v) for v in reused_by_id.values())
            total_needed = sum(need_by_id.values())
            if existing_file:
                print(f"Resume mode: reused {total_skipped} existing responses; need {total_needed} new generations")
            
            new_by_id: Dict[str, List[str]] = {pid: [] for pid in need_by_id.keys()}
            prompts_by_need: Dict[int, List[str]] = defaultdict(list)
            for pid, need_cnt in need_by_id.items():
                if need_cnt > 0:
                    prompts_by_need[need_cnt].append(pid)
            
            # Interpret non-positive max_tokens as very large number for vLLM (acts as practically unlimited)
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
                combined = (reused + newly)[:num_responses]
                while len(combined) < num_responses:
                    combined.append("[ERROR: Response not generated]")
                result_entry = {
                    "id": prompt_data['task_id'],
                    "prompt": prompt_data['metadata_prompt'],
                    "responses": combined
                }
                # Include health data if present
                if 'health_data' in prompt_data:
                    result_entry.update(prompt_data['health_data'])
                results_to_save.append(result_entry)
        
        # Save results
        save_results(results_to_save, output_file)
        
        # Clean up checkpoint file
        cleanup_checkpoint(output_file)
        
        print(f"Rollout completed. Saved {len(results_to_save)} prompts to {output_file}")
        
        # Clean up vLLM instance if we created it (only for non-API mode)
        if not use_api and enable_cleanup and vllm_instance is None and 'llm' in locals():
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
            
            # Clear any remaining CUDA contexts
            try:
                import nvidia_ml_py as nvml
                nvml.nvmlInit()
                for i in range(torch.cuda.device_count()):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    nvml.nvmlDeviceResetApplicationsClocks(handle)
                nvml.nvmlShutdown()
            except:
                # nvidia-ml-py might not be installed
                pass
                
            print("Cleanup completed.")
        elif not enable_cleanup:
            print("Skipping cleanup (--no-cleanup flag is set)")
            
        return True
        
    except Exception as e:
        print(f"Error in generate_responses_api: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
