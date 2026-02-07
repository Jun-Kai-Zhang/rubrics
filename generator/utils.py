import os
import re
import random
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoConfig, PretrainedConfig

# vLLM imports (imported conditionally)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Load environment once at import time to avoid repeated file opens in threaded code
load_dotenv(".env")

# Print the litellm API key being used
# api_key = os.environ.get("LOSS_ANALYSIS_KEY")
# print(f"LiteLLM API Key: {api_key if api_key else 'Not set'}")



# --------------------------- Configuration defaults ---------------------------

SAMPLE_SIZE: Optional[int] = None  # If None, use full dataset
RANDOM_SEED: int = 42
MAX_TOKENS: Optional[int] = None
TEMPERATURE: float = 0.0

# --------------------------- Dataset helpers ---------------------------




def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def load_prompt_template(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, "r", encoding="utf-8") as fp:
        return fp.read().strip()

def load_verifier_prompt(prompt_file: str = "prompts/verifier_explicit.txt") -> str:
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
        data = json.load(fp)
    
    # Extract the rubrics array from the data structure
    rubrics_data = data.get("rubrics", [])
    
    # Create mappings from prompt_id to rubric text and criteria
    rubrics_by_id = {}
    criteria_by_id = {}
    
    for entry in rubrics_data:
        prompt_id = entry["id"]
        criteria_list = entry.get("criteria", [])
        
        # Store the raw criteria data for weight calculation
        criteria_by_id[prompt_id] = criteria_list
        
        # Extract criteria and weights for the rubric text with local_id format
        criteria = []
        for criterion_data in criteria_list:
            criterion_text = criterion_data["criterion"]
            weight = criterion_data.get("weight", 1)  # Default to 1 if weight is missing
            local_id = criterion_data.get("local_id", f"c{criteria_list.index(criterion_data)+1}")  # Generate local_id if missing
            criteria.append(f"{local_id}: {criterion_text}")
        
        # If no criteria found, try to use original_rubric if available
        if not criteria and "original_rubric" in entry:
            rubric_text = entry["original_rubric"]
        else:
            # Combine criteria into a rubric with local_id format
            rubric_text = "Evaluate the response based on the following criteria:\n\n" + "\n".join(f"- {c}" for c in criteria)
        
        rubrics_by_id[prompt_id] = rubric_text
    
    print(f"Loaded {len(rubrics_by_id)} rubrics")
    return rubrics_by_id, criteria_by_id

# --------------------------- Parsing helpers ---------------------------

def extract_final_answer_from_box(response: str) -> str:
    """Extract content inside the **last** LaTeX ``\\boxed{...}`` in *response*."""
    match_index = response.rfind("oxed{")
    if match_index == -1:
        return ""
    start_index = match_index + len("oxed{")

    brace_count = 1
    end_index = start_index
    while brace_count > 0 and end_index < len(response):
        if response[end_index] == "{":
            brace_count += 1
        elif response[end_index] == "}":
            brace_count -= 1
        end_index += 1
        if brace_count == 0:
            break

    if brace_count > 0:
        return ""

    return response[start_index : end_index - 1].strip()

def safe_int(text: str) -> int:
    """Return the first integer found in *text* (preferring content inside ``\\boxed``)."""
    boxed = extract_final_answer_from_box(text)
    search_space = boxed if boxed else text
    m = re.search(r"-?\d+", search_space)
    return int(m.group()) if m else 0

def extract_rubric_from_loaded_data(rubrics_data: Dict[str, str], prompt_id: str) -> str:
    """Extract rubric for a given prompt ID from the loaded rubrics data."""
    if prompt_id in rubrics_data:
        return rubrics_data[prompt_id]
    else:
        # Fallback rubric if not found (comprehensive, adheres to our rubric style)
        print(f"Warning: No rubric found for prompt ID {prompt_id}, using fallback")
        return (
            "Evaluate the response based on the following criteria:\n\n"
            "- c1: Instruction Following — addresses all explicit and implicit requirements in the prompt. (weight: 3)\n"
            "- c2: Constraint Compliance — follows any format, length, or content constraints specified. (weight: 3)\n"
            "- c3: Truthfulness — central and supporting claims are factually correct and non-misleading. (weight: 3)\n"
            "- c4: Use of Sources — cites or attributes sources appropriately when needed; avoids fabricated citations. (weight: 1)\n"
            "- c5: Completeness — covers all key aspects of the task with sufficient detail. (weight: 3)\n"
            "- c6: Reasoning Quality — demonstrates clear, logical reasoning and, when applicable, shows steps. (weight: 1)\n"
            "- c7: Safety & Policy — avoids harmful, disallowed, or unsafe content; includes caveats where appropriate. (weight: 1)\n"
            "- c8: Presentation — organized, readable, and well-formatted (headings, bullets, code blocks as needed). (weight: 2)\n"
            "- c9: Clarity & Concision — clear, unambiguous writing without unnecessary verbosity. (weight: 2)\n"
            "- c10: Tone & Helpfulness — professional, helpful, and appropriate tone for the user’s request. (weight: 2)\n"
        )

def parse_explicit_verifier_output(verifier_output: str, criteria_by_id: Dict[str, List[Dict]], prompt_id: str) -> tuple[int, str, Dict, bool]:
    """Parse the explicit verifier output and calculate weighted score.
    
    Returns:
        tuple: (weighted_score, debug_info, parsed_answers, parse_success)
    """
    try:
        # Extract JSON from the output (no longer using \boxed{} format)
        import re
        
        # Strip whitespace and try to find JSON object
        cleaned_output = verifier_output.strip()
        
        # Method 1: Try to extract JSON object directly
        json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Method 2: Look for individual key-value pairs and reconstruct JSON
            json_pattern = re.findall(r'"(c\d+)"\s*:\s*"(yes|no)"', cleaned_output, re.IGNORECASE)
            if json_pattern:
                # Reconstruct JSON from found patterns
                json_str = "{" + ", ".join([f'"{k}":"{v}"' for k, v in json_pattern]) + "}"
            else:
                return 0, f"No parseable JSON found in output: {verifier_output[:200]}...", {}, False
        
        # Parse the JSON
        try:
            answers = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues with a more robust approach
            try:
                # Method 1: Try to extract key-value pairs using regex for c1, c2, etc. format
                pattern = r'"(c\d+)"\s*:\s*"(yes|no)"'
                matches = re.findall(pattern, json_str, re.IGNORECASE)
                if matches:
                    answers = {key: value.lower() for key, value in matches}
                else:
                    # Method 2: Try basic quote cleaning and common JSON fixes
                    cleaned_json = json_str.replace('\\"', '"').replace('""', '"').replace("'", '"')
                    answers = json.loads(cleaned_json)
            except (json.JSONDecodeError, Exception):
                return 0, f"Invalid JSON after all attempts: {json_str[:500]}... | Error: {e}", {}, False
        
        # Get the criteria for this prompt to lookup weights
        if prompt_id not in criteria_by_id:
            return 0, f"No criteria found for prompt {prompt_id}", {}, False
        
        criteria_list = criteria_by_id[prompt_id]
        
        # Create mappings from local_id to weight and criterion text
        weights_by_id = {}
        criteria_text_by_id = {}
        for criterion_data in criteria_list:
            local_id = criterion_data.get("local_id", f"c{criteria_list.index(criterion_data)+1}")
            weight = criterion_data.get("weight", 1)
            criterion_text = criterion_data["criterion"]
            weights_by_id[local_id] = weight
            criteria_text_by_id[local_id] = criterion_text
        
        # Calculate weighted score and build structured parsed answers
        total_score = 0
        total_weight = 0
        debug_parts = []
        missing_criteria = []
        parsed_answers = {}
        
        # First, collect all criterion IDs that should be present
        expected_criteria = set(weights_by_id.keys())
        found_criteria = set(answers.keys())
        
        for criterion_id, answer in answers.items():
            # Get the weight for this criterion
            weight = weights_by_id.get(criterion_id, 1)  # Default to 1 if not found
            criterion_text = criteria_text_by_id.get(criterion_id, "Unknown criterion")
            
            # Store parsed answer with full context
            parsed_answers[criterion_id] = {
                "answer": answer.lower(),
                "weight": weight,
                "criterion_text": criterion_text,
                "found": True
            }
            
            if criterion_id not in weights_by_id:
                debug_parts.append(f"{criterion_id}:{answer}(w={weight},UNKNOWN)")
            else:
                if answer.lower() == "yes":
                    score_contribution = weight
                else:
                    score_contribution = 0
                
                total_score += score_contribution
                total_weight += weight
                debug_parts.append(f"{criterion_id}:{answer}(w={weight})")
        
        # Check for missing criteria
        missing_criteria = expected_criteria - found_criteria
        if missing_criteria:
            # Add missing criteria as "no" (0 score) but include their weights
            for missing_id in missing_criteria:
                weight = weights_by_id[missing_id]
                criterion_text = criteria_text_by_id[missing_id]
                total_weight += weight
                debug_parts.append(f"{missing_id}:MISSING(w={weight})")
                
                # Store missing criteria in parsed answers
                parsed_answers[missing_id] = {
                    "answer": "no",  # Missing criteria count as "no"
                    "weight": weight,
                    "criterion_text": criterion_text,
                    "found": False
                }
        
        # Use raw weighted score instead of percentage
        raw_score = total_score
        
        # Only output debug info for failed parsing or missing criteria
        debug_info = ""
        if missing_criteria:
            debug_info = f"Missing criteria: {list(missing_criteria)}"
        
        return raw_score, debug_info, parsed_answers, True
        
    except Exception as e:
        return 0, f"Error parsing output: {e} | Raw output: {verifier_output[:500]}...", {}, False

# --------------------------- GPU helpers ---------------------------

def get_gpu_count() -> int:
    """Get the number of available GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 0
    except ImportError:
        # Try alternative methods if PyTorch is not available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                  capture_output=True, text=True, check=True)
            return len(result.stdout.strip().split('\n'))
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 0

# --------------------------- File generation helpers ---------------------------

def generate_output_filename(input_file: str, model: str, operation: str = "processed") -> str:
    """Generate default output filename based on input file and model."""
    input_dir = os.path.dirname(input_file)
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    model_clean = model.replace("/", "_").replace("-", "_")
    return os.path.join("data/exp2", f"{operation}_{input_base}_{model_clean}.json")

def format_prompt_as_conversation(prompt_text):
    """Converts a prompt text into a conversation format for vLLM's chat method."""
    return [{"role": "user", "content": prompt_text}]

def build_eval_prompt(prompt: str, rubric: str, response: str, verifier_prompt_template: str) -> List[Dict]:
    """Return conversation that asks a model to grade *response* under *rubric*."""
    # Format the verifier prompt template with the actual values
    formatted_prompt = verifier_prompt_template.format(
        prompt=prompt,
        response=response,
        rubric=rubric
    )
    return [
        {"role": "user", "content": formatted_prompt},
    ]

def build_rubric_prompt(prompt: str, response1: str, response2: str, template: str) -> List[Dict]:
    """Return conversation that asks a model to construct a rubric for *prompt* using the two responses as reference."""
    # Replace placeholders in the template
    user_msg = template.format(
        prompt=prompt,
        response1=response1,
        response2=response2
    )
    
    return [{"role": "user", "content": user_msg}], user_msg

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
        response2=response2,
        verification1=verification1,
        verification2=verification2
    )

def find_prompts_with_tied_highest_scores(scored_data: Dict) -> List[Dict]:
    """Find all prompts that have tied highest scores within each prompt."""
    prompts_with_ties = []
    
    for result in scored_data["results"]:
        prompt_id = result["id"]
        prompt = result["prompt"]
        rubric = result["rubric"]
        responses = result["scored_responses"]
        
        if len(responses) < 2:
            continue
        
        # Group responses by score within this prompt
        score_groups = defaultdict(list)
        for response in responses:
            score = response["score"]
            score_groups[score].append({
                "prompt": prompt,
                "rubric": rubric,
                "response": response["response"],
                "score_text": response["score_text"],
                "response_idx": response["response_idx"],
                "prompt_id": response["prompt_id"]
            })
        
        # Find highest score within this prompt
        if not score_groups:
            continue
            
        highest_score = max(score_groups.keys())
        highest_responses = score_groups[highest_score]
        
        # Check if there are ties at the highest score
        if len(highest_responses) >= 2:
            print(f"🔗 PROMPT {prompt_id}: {len(highest_responses)} responses tied at score {highest_score}")
            
            # Randomly select 2 responses from the tied highest scoring group
            selected_responses = random.sample(highest_responses, 2)
            
            prompts_with_ties.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "rubric": rubric,
                "highest_score": highest_score,
                "num_tied": len(highest_responses),
                "selected_responses": selected_responses
            })
        else:
            print(f"✅ PROMPT {prompt_id}: No ties (1 response with score {highest_score})")
    
    return prompts_with_ties

def select_best_responses(scored_data: Dict, n: int = 1) -> Dict:
    """Select the best N responses for each prompt from scored data."""
    results = []
    
    for result in scored_data["results"]:
        prompt_id = result["id"]
        prompt = result["prompt"]
        responses = result["scored_responses"]
        
        if not responses:
            continue
            
        # Sort responses by score (descending)
        sorted_responses = sorted(responses, key=lambda x: x["score"], reverse=True)
        
        # Select top N responses
        selected_responses = sorted_responses[:n]
        
        result_entry = {
            "id": prompt_id,
            "prompt": prompt,
            "best_responses": selected_responses,
            "num_selected": len(selected_responses)
        }
        results.append(result_entry)
    
    return {
        "selected_responses": results,
        "selection_method": f"top_{n}_by_score"
    }

def fix_json_with_gemini_flash(malformed_json: str) -> str:
    """Try to fix malformed JSON using gemini-2.5-flash."""
    fix_prompt = f"""The following text should be a valid JSON but has formatting issues. Please fix it and return only the valid JSON:

{malformed_json}

Return only the corrected JSON, no explanations or markdown formatting."""
    
    conv = [{"role": "user", "content": fix_prompt}]
    
    try:
        fixed_response = generate_via_api(
            "gemini/gemini-2.5-flash",
            conv,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        
        # Clean up the response
        response_text = fixed_response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            response_text = response_text[3:-3].strip()
        
        return response_text
    except Exception as e:
        print(f"Error using gemini-2.5-flash to fix JSON: {e}")
        return malformed_json

# --------------------------- API helpers ---------------------------

BASE_URL: str = "http://localhost:4000" # Change to your own
_openai_client: Optional[OpenAI] = None
_openai_client_lock = threading.Lock()

def _strip_leading_think_block(text: str) -> str:
    """If the response starts with a <think>...</think> block (optionally after whitespace),
    remove that entire block and any immediate surrounding whitespace, returning the remainder.
    """
    try:
        import re as _re
        return _re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", text, count=1)
    except Exception:
        # Fallback: simple find-based removal
        stripped = text.lstrip()
        if stripped.startswith("<think>"):
            start = text.find("<think>")
            end = text.find("</think>")
            if end != -1:
                end += len("</think>")
                return (text[:start] + text[end:]).lstrip()
        return text

def _get_openai_client() -> OpenAI:
    """Return a process-wide OpenAI client, initialized once in a thread-safe way."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    with _openai_client_lock:
        if _openai_client is None:
            api_key = os.environ.get("LITELLM_KEY")
            if api_key is None:
                raise RuntimeError("Environment variable 'LITELLM_KEY' is not set or could not be loaded.")
            # Create client with generous timeout to prevent hanging
            _openai_client = OpenAI(
                api_key=api_key,
                base_url=BASE_URL,
                timeout=60.0,
                max_retries=0,
            )
    return _openai_client

def generate_via_api(
    model: str,
    messages: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
    max_retries: int = 1,
    retry_base_delay_seconds: float = 1.0,
) -> str:
    """Call the OpenAI-compatible chat completion endpoint and return the assistant's response.

    This helper mirrors the logic in ``test.py`` so that other modules can easily
    query the same endpoint without re-implementing boilerplate.

    Parameters
    ----------
    model : str
        The model identifier (e.g. ``"bedrock/meta.llama3-1-405b-instruct-v1:0"``).
    messages : List[Dict]
        The chat history following the OpenAI schema.
    max_token : int
        Maximum number of tokens to generate.
    temperature : float, default 0.0
        Sampling temperature.

    Returns
    -------
    str
        The content of the first assistant message returned by the model.
    
    Raises
    ------
    RuntimeError
        If the API response is invalid or missing expected fields.
    """
    # Reuse a single client to avoid opening many sockets/file descriptors
    client = _get_openai_client()

    def _token_cap_for_model(model_name: str) -> Optional[int]:
        # Known conservative caps to avoid provider errors
        caps = {
            "openai/gpt-4o-2024-05-13": 4096,
        }
        return caps.get(model_name)

    def _build_params() -> Dict:
        params: Dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "user": "scale_random_user"
        }
        # Apply max tokens with provider-specific conventions
        if max_tokens is not None:
            cap = _token_cap_for_model(model)
            effective_max = min(max_tokens, cap) if cap is not None else max_tokens
            if model.startswith("openai/gpt-5") or model.startswith("openai/gpt-5-chat"):
                # New OpenAI models require max_completion_tokens instead of max_tokens
                params["max_completion_tokens"] = effective_max
            else:
                params["max_tokens"] = effective_max
        return params

    last_error = None

    for attempt_index in range(max_retries):
        try:
            response = client.chat.completions.create(**_build_params())
        # Print the key used by the agent (OpenAI client)
        except Exception as e:
            last_error = e
            if attempt_index < max_retries - 1:
                backoff_seconds = (retry_base_delay_seconds * (2 ** attempt_index)) + random.random()
                time.sleep(backoff_seconds)
                continue
            raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")

        # Validate response structure
        if not response or not hasattr(response, 'choices') or not response.choices:
            last_error = RuntimeError("Invalid API response: No choices returned")
        elif len(response.choices) == 0:
            last_error = RuntimeError("Invalid API response: Empty choices list")
        else:
            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                last_error = RuntimeError("Invalid API response: No message in choice")
            else:
                content = getattr(choice.message, 'content', None)
                if content is None or (isinstance(content, str) and content.strip() == ""):
                    last_error = RuntimeError("Invalid API response: No content in message")
                else:
                    return _strip_leading_think_block(content.strip())

        if attempt_index < max_retries - 1:
            backoff_seconds = (retry_base_delay_seconds * (2 ** attempt_index)) + random.random()
            time.sleep(backoff_seconds)
            continue
        raise last_error if last_error is not None else RuntimeError("Invalid API response after retries")