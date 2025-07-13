# llm_handler.py
import requests
import json
from config import OLLAMA_API_ENDPOINT, MISTRAL_MODEL_NAME

DEFAULT_SYSTEM_PROMPT = """You are a specialized assistant for a rail operations company.
Your ONLY function is to answer questions based SOLELY on the context provided with each query related to rail system health codes, vehicle data, causal analysis summaries, and predictive maintenance reports.
Do not use any external knowledge or make assumptions beyond this context.
You MUST REFUSE to answer any question that is outside the rail operations domain (e.g., general knowledge, jokes, weather, other industries) or if the provided context is insufficient. State clearly that the request is outside your scope or that you lack the specific information. DO NOT attempt to be creative or fulfill out-of-domain requests in any way.
Be concise and stick strictly to the information given.
"""

def call_mistral_model(prompt, system_message=DEFAULT_SYSTEM_PROMPT, context_data=""):
    """
    Calls the local Mistral model via Ollama API.
    'context_data' is pre-fetched information relevant to the user's query.
    """
    full_prompt = f"{context_data}\n\nUser Question: {prompt}\n\nAssistant:"
    
    # --- DEBUGGING: Print prompt length and part of the prompt ---
    print(f"DEBUG_LLM: Full prompt length: {len(full_prompt)}")
    print(f"DEBUG_LLM: Full prompt (first 500 chars):\n{full_prompt[:500]}\n...")
    # --- END DEBUGGING ---

    payload = {
        "model": MISTRAL_MODEL_NAME,
        "prompt": full_prompt,
        "stream": False, 
        "system": system_message
    }
    try:
        print(f"DEBUG_LLM: Sending request to Ollama: {OLLAMA_API_ENDPOINT} with model {MISTRAL_MODEL_NAME}")
        response = requests.post(OLLAMA_API_ENDPOINT, json=payload, timeout=180) # Increased timeout further
        
        # --- DEBUGGING: Print status code and raw response text ---
        print(f"DEBUG_LLM: Ollama response status code: {response.status_code}")
        print(f"DEBUG_LLM: Ollama raw response text (first 500 chars): {response.text[:500]}")
        # --- END DEBUGGING ---

        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        response_data = response.json()
        
        if 'response' in response_data:
            return response_data['response'].strip()
        elif 'message' in response_data and 'content' in response_data['message']:
             return response_data['message']['content'].strip()
        else: 
            print(f"DEBUG_LLM: Unexpected Ollama JSON structure: {response_data}")
            # Try a more generic way to find text if standard keys fail
            if isinstance(response_data, dict):
                for key, value in response_data.items():
                    if isinstance(value, str) and len(value.strip()) > 10 : # Try to find a meaningful string
                        print(f"DEBUG_LLM: Fallback - using value from key '{key}' as response.")
                        return value.strip()
            return f"Received an unexpected response structure from Ollama. Full response: {str(response_data)[:500]}"

    except requests.exceptions.Timeout:
        print(f"Error calling Ollama API: Request timed out after 180 seconds.")
        return "Sorry, the request to the language model timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return "Sorry, I encountered an error trying to reach the language model."
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from Ollama. Raw response: {response.text[:1000]}") # Log more of raw response
        return "Sorry, I received an unreadable response from the language model."
    except Exception as e:
        print(f"An unexpected error occurred in call_mistral_model: {e}")
        return "Sorry, an unexpected error occurred while processing your request with the language model."