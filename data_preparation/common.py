"""Common utilities for data preparation scripts.

This module provides shared functionality for data preparation including:
- Loading prompts from various file formats
- Managing existing results for resume capability
- Checkpoint management
- Standardized output file naming

Supported Input Formats:
1. training_prompts.json - List of objects with 'prompt' field
   Example: [{"prompt": "Write a story..."}, ...]
   
2. CSV files - Must have 'task_id' and 'metadata_prompt' columns
   Example: task_id,metadata_prompt
            1,"Generate code for..."
            
3. Standard JSON - List of objects with 'id' and 'prompt' fields
   Example: [{"id": "task1", "prompt": "Explain..."}, ...]
   
4. Other supported JSON fields:
   - 'conversation_id' and 'instruction' (wildchat format)
   - 'ids'/'id' and 'prompts'/'prompt' (flexible field names)
   - Health data format with additional 'criteria_*' fields
"""
import json
import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import random
from pathlib import Path
import sys

# Load environment variables from the specified .env file
from dotenv import load_dotenv
env_path = ".env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"Warning: Environment file not found at {env_path}")

# Try to import from generator.utils
try:
    from generator.utils import generate_via_api
except ImportError:
    # Add parent directory to path and try again
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from generator.utils import generate_via_api
    except ImportError:
        # If still failing, define a placeholder
        print("Warning: Could not import generate_via_api from generator.utils")
        generate_via_api = None


def format_prompt_as_conversation(prompt_text: str) -> List[Dict[str, str]]:
    """
    Converts a prompt text into a conversation format for API calls.
    
    Args:
        prompt_text: The prompt text to convert
        
    Returns:
        List containing a single message dict with role="user"
    """
    return [{"role": "user", "content": prompt_text}]


def load_prompts_from_file(
    input_file: str,
    sample_size: Optional[int] = None,
    sequential: bool = False,
    random_seed: int = 42
) -> List[Dict[str, str]]:
    """
    Load prompts from a JSON or CSV file with consistent handling.
    
    Supports multiple formats:
    - training_prompts.json: List of {prompt: str} objects
    - CSV files: Must have 'task_id' and 'metadata_prompt' columns
    - Other JSON formats with various field names
    
    Args:
        input_file: Path to the input file
        sample_size: Number of prompts to sample (None for all)
        sequential: If True, take first N prompts; if False, sample randomly
        random_seed: Random seed for sampling
        
    Returns:
        List of dicts with 'task_id' and 'metadata_prompt' fields
    """
    random.seed(random_seed)
    
    if not input_file or not os.path.exists(input_file):
        raise ValueError(f"Input file not found: {input_file}")
    
    prompts_data = []
    
    if input_file.endswith('.json'):
        with open(input_file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        
        # Handle training_prompts.json format: list of {prompt: str}
        if isinstance(data, list) and data and isinstance(data[0], dict):
            if 'prompt' in data[0] and 'id' not in data[0]:
                # This is training_prompts.json format
                print(f"Detected training_prompts.json format with {len(data)} prompts")
                if sample_size:
                    if sequential:
                        data = data[:sample_size]
                        print(f"Using sequential loading: first {len(data)} prompts")
                    else:
                        data = random.sample(data, min(sample_size, len(data)))
                        print(f"Using random sampling: {len(data)} prompts")
                
                prompts_data = [
                    {
                        'task_id': f"prompt_{idx}",
                        'metadata_prompt': item.get('prompt', '')
                    }
                    for idx, item in enumerate(data)
                ]
            else:
                # Handle other JSON formats with id/prompt fields
                print(f"Detected JSON format with {len(data)} prompts")
                if sample_size:
                    if sequential:
                        data = data[:sample_size]
                    else:
                        data = random.sample(data, min(sample_size, len(data)))
                
                # Try different field names for ID and prompt
                for idx, item in enumerate(data):
                    task_id = (
                        item.get('id') or 
                        item.get('task_id') or 
                        item.get('conversation_id') or 
                        f"json_{idx}"
                    )
                    prompt = (
                        item.get('prompt') or 
                        item.get('metadata_prompt') or 
                        item.get('instruction') or 
                        item.get('prompts') or 
                        ''
                    )
                    prompts_data.append({
                        'task_id': str(task_id),
                        'metadata_prompt': prompt
                    })
        else:
            raise ValueError(f"Unsupported JSON format in {input_file}")
            
    elif input_file.endswith('.csv'):
        print(f"Loading prompts from CSV: {input_file}")
        df = pd.read_csv(input_file, low_memory=False)
        
        # Check for required columns
        if 'task_id' not in df.columns or 'metadata_prompt' not in df.columns:
            raise ValueError("CSV must have 'task_id' and 'metadata_prompt' columns")
        
        if sample_size:
            if sequential:
                df = df.head(sample_size)
                print(f"Using sequential loading: first {len(df)} prompts")
            else:
                df = df.sample(min(sample_size, len(df)), random_state=random_seed)
                print(f"Using random sampling: {len(df)} prompts")
        
        prompts_data = df[['task_id', 'metadata_prompt']].to_dict('records')
        # Ensure task_id is string
        for item in prompts_data:
            item['task_id'] = str(item['task_id'])
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    print(f"Loaded {len(prompts_data)} prompts")
    return prompts_data


def save_results(
    results: List[Dict],
    output_file: str,
    ensure_ascii: bool = False
) -> None:
    """
    Save results to a JSON file with proper formatting.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output file
        ensure_ascii: Whether to escape non-ASCII characters
    """
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=ensure_ascii)
    print(f"Saved {len(results)} results to {output_file}")


def build_output_filename(
    output_dir: str,
    model_name: str,
    temperature: float,
    top_p: float,
    num_prompts: int,
    num_responses: int,
    label: str = "Responses"
) -> str:
    """
    Build a standardized output filename.
    
    Args:
        output_dir: Output directory
        model_name: Model identifier
        temperature: Temperature parameter
        top_p: Top-p parameter
        num_prompts: Number of prompts
        num_responses: Number of responses per prompt
        label: Additional label for the filename
        
    Returns:
        Full path to output file
    """
    # Clean model name for filename
    clean_model = model_name.split("/")[-1].replace("/", "_").replace("-", "_")
    filename = (
        f"Policy_Model_{clean_model}_"
        f"Temperature_{temperature}_TopP_{top_p}_"
        f"{num_prompts}_Prompts_{num_responses}_Responses_{label}.json"
    )
    return os.path.join(output_dir, filename)


def load_existing_results(
    existing_file: str,
    is_multi_model: bool = False
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, List]]]:
    """
    Load existing results from a file for resume functionality.
    
    Args:
        existing_file: Path to existing results JSON
        is_multi_model: If True, expects responses to have 'generator' field
        
    Returns:
        Tuple of:
        - existing_by_id: Dict mapping task_id to list of responses
        - existing_by_prompt_text: Dict mapping prompt text to list of responses
        - existing_by_prompt_and_model: Dict mapping task_id to model to list of responses
          (only populated if is_multi_model=True)
    """
    existing_by_id = {}
    existing_by_prompt_text = {}
    existing_by_prompt_and_model = {}
    
    if not existing_file or not os.path.exists(existing_file):
        return existing_by_id, existing_by_prompt_text, existing_by_prompt_and_model
    
    try:
        with open(existing_file, "r", encoding="utf-8") as fp:
            existing_data = json.load(fp)
        
        if not isinstance(existing_data, list):
            print(f"Warning: existing file is not a list, skipping resume")
            return existing_by_id, existing_by_prompt_text, existing_by_prompt_and_model
        
        for entry in existing_data:
            if not isinstance(entry, dict):
                continue
            
            task_id = entry.get("id") or entry.get("task_id")
            prompt_text = entry.get("prompt")
            raw_responses = entry.get("responses", [])
            
            if is_multi_model:
                # Multi-model format: responses have 'generator' field
                by_model = {}
                for resp in raw_responses:
                    if isinstance(resp, dict) and "generator" in resp:
                        model_name = resp.get("generator")
                        by_model.setdefault(model_name, []).append(resp)
                    else:
                        # Unlabeled response
                        by_model.setdefault("__unlabeled__", []).append(resp)
                
                if task_id:
                    existing_by_prompt_and_model[str(task_id)] = by_model
            else:
                # Single-model format: responses are strings or dicts with 'response' field
                responses = []
                for r in raw_responses:
                    if isinstance(r, dict):
                        val = r.get("response")
                        if isinstance(val, str):
                            responses.append(val)
                    elif isinstance(r, str):
                        responses.append(r)
                
                if task_id:
                    existing_by_id[str(task_id)] = responses
                if prompt_text and prompt_text.strip():
                    existing_by_prompt_text[prompt_text] = responses
        
        print(f"Loaded existing results from {existing_file}")
        if is_multi_model:
            total_existing = sum(
                len(responses) 
                for by_model in existing_by_prompt_and_model.values() 
                for responses in by_model.values()
            )
            print(f"Found {total_existing} existing responses across {len(existing_by_prompt_and_model)} prompts")
        else:
            total_existing = sum(len(v) for v in existing_by_id.values())
            print(f"Found {total_existing} existing responses across {len(existing_by_id)} prompts")
            
    except Exception as ex:
        print(f"Warning: Failed to load existing file '{existing_file}': {ex}")
    
    return existing_by_id, existing_by_prompt_text, existing_by_prompt_and_model


def save_checkpoint(
    output_file: str,
    results: List[Dict],
    checkpoint_suffix: str = ".tmp"
) -> bool:
    """
    Save a checkpoint of current results.
    
    Args:
        output_file: Base output filename
        results: Current results to save
        checkpoint_suffix: Suffix to append to filename for checkpoint
        
    Returns:
        True if successful, False otherwise
    """
    try:
        checkpoint_file = output_file + checkpoint_suffix
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Checkpoint saved: {len(results)} results → {checkpoint_file}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")
        return False


def cleanup_checkpoint(output_file: str, checkpoint_suffix: str = ".tmp") -> None:
    """
    Remove checkpoint file if it exists.
    
    Args:
        output_file: Base output filename
        checkpoint_suffix: Suffix used for checkpoint file
    """
    checkpoint_file = output_file + checkpoint_suffix
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"Removed checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")
