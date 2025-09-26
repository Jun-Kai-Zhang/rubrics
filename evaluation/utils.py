import os
from typing import List, Dict, Optional, Tuple


from dotenv import load_dotenv
from openai import OpenAI



# --------------------------- Dataset helpers ---------------------------

SAMPLE_SIZE: Optional[int] = None  # If None, use full dataset
RANDOM_SEED: int = 42

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



# --------------------------- API helpers ---------------------------

def generate_via_api(
    model: str,
    messages: List[Dict],
    max_tokens: Optional[int] = None,
    temperature: float = 0.0,
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
    # Load credentials from the workspace-level .env (matches behaviour in ``test.py``)
    load_dotenv(".env")

    base_url = "Your base url"
    api_key = os.environ.get("API_KEY")
    if api_key is None:
        raise RuntimeError("Environment variable 'API_KEY' is not set or could not be loaded.")

    # Create client with 5 minute timeout to prevent hanging
    # This timeout applies to the entire request/response cycle
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=300.0  # 5 minutes timeout
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")

    # Validate response structure
    if not response or not hasattr(response, 'choices') or not response.choices:
        raise RuntimeError(f"Invalid API response: No choices returned")
    
    if len(response.choices) == 0:
        raise RuntimeError(f"Invalid API response: Empty choices list")
    
    choice = response.choices[0]
    if not hasattr(choice, 'message') or not choice.message:
        raise RuntimeError(f"Invalid API response: No message in choice")
    
    if not hasattr(choice.message, 'content') or choice.message.content is None:
        raise RuntimeError(f"Invalid API response: No content in message")
    
    # The Scale proxy returns completions in standard OpenAI format.
    return choice.message.content.strip()