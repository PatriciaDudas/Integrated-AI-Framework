# knowledge_base.py
import re
from config import CAUSAL_SUMMARY_FILE

CACHED_CAUSAL_SUMMARIES_DICT = {} # Stores parsed summaries

def _create_simplified_key(question_title):
    """Helper to create a simplified key from a question title for matching."""
    key_part = question_title.lower()
    # Remove common phrases and punctuation that don't help identify the core question
    key_part = re.sub(r"does operational mode|does entering|vs|cause an increase in the probability of|cause|events|\(|\)|'|,|\?|\.", "", key_part)
    key_part = re.sub(r"health codes|healthcode|hc", "hc", key_part) # Normalize healthcode terms
    key_part = re.sub(r"track sections|tracksection|track section|section", "ts", key_part) # Normalize tracksection terms
    key_part = re.sub(r"\s+", " ", key_part).strip() # Normalize spaces
    
    # Extract core entities (types, HCs, TS numbers, keywords like "slip slide")
    tokens = key_part.split()
    entities = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "type" and i + 1 < len(tokens) and tokens[i+1].isalnum():
            entities.append(f"type {tokens[i+1]}")
            i += 1 
        elif token == "hc" and i + 1 < len(tokens) and tokens[i+1].isalnum():
            entities.append(f"hc {tokens[i+1]}")
            i += 1
        elif token == "ts" and i + 1 < len(tokens) and tokens[i+1].isdigit(): # Assuming TS IDs are digits
            entities.append(f"ts {tokens[i+1]}")
            i += 1
        elif token in ["slip", "slide"]:
            if "slip slide" not in entities: entities.append("slip slide")
        elif token == "door" and "failures" in tokens: # check for "door failures"
            if "door failures" not in entities: entities.append("door failures")
        elif token == "simulated" and "transponders" in tokens:
             if "simulated transponders" not in entities: entities.append("simulated transponders")
        elif not token.isdigit() and token not in ["type", "hc", "ts", "and", "or", "the", "a", "is", "to"]: # Avoid standalone numbers and common small words
             entities.append(token)
        i += 1
        
    return " ".join(sorted(list(set(entities)))) # Sorted for canonical key


def parse_and_load_causal_summary():
    """
    Parses the causal analysis summary file into a dictionary.
    Keys are simplified question identifiers, values are the summary texts.
    """
    global CACHED_CAUSAL_SUMMARIES_DICT
    if CACHED_CAUSAL_SUMMARIES_DICT and "error" not in CACHED_CAUSAL_SUMMARIES_DICT : # Already loaded and parsed, no error
        return

    raw_content = ""
    try:
        with open(CAUSAL_SUMMARY_FILE, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        print(f"Causal summary raw content loaded from {CAUSAL_SUMMARY_FILE}")
    except FileNotFoundError:
        error_msg = f"Error: Causal summary file not found at {CAUSAL_SUMMARY_FILE}"
        print(error_msg)
        CACHED_CAUSAL_SUMMARIES_DICT = {"error": error_msg}
        return
    except Exception as e:
        error_msg = f"Error loading causal summary: {e}"
        print(error_msg)
        CACHED_CAUSAL_SUMMARIES_DICT = {"error": error_msg}
        return

    # Split the content into blocks. Assume each question starts with "Does "
    # and is separated by at least two blank lines (effectively 3 newlines or more if lines have trailing spaces).
    # This regex splits on two or more newlines that are followed by "Does "
    question_blocks = re.split(r'\n\s*\n(?=Does )', raw_content)
    
    # The first block might not start with "Does " if the file doesn't begin that way
    # or if the first question is the only content.
    if not raw_content.strip().startswith("Does ") and question_blocks:
        # If the first block isn't a question, it might be an intro or needs different handling
        # For now, we assume the first block *is* the first question if file starts with "Does"
        # or it's the only block.
         if not question_blocks[0].strip().startswith("Does "):
             first_block_is_question = False
             if len(question_blocks) == 1: # If only one block, assume it's the relevant summary
                 pass # We'll handle this below
             else: # Prepend "Does " to subsequent blocks if split removed it
                 for i in range(1, len(question_blocks)):
                     question_blocks[i] = "Does " + question_blocks[i]
         
    elif raw_content.strip().startswith("Does ") and len(question_blocks)>0 and not question_blocks[0].strip().startswith("Does "):
        # This handles the case where re.split might consume the "Does " for the first element
        # if the file starts directly with "Does " and the split pattern affects the very beginning.
        question_blocks[0] = "Does " + question_blocks[0]


    parsed_summaries_temp = {}
    for i, block_text in enumerate(question_blocks):
        block_text = block_text.strip()
        if not block_text:
            continue

        try:
            # Assume the first line of a block is the question title
            parts = block_text.split('\n', 1)
            question_title = parts[0].strip()
            summary_content = parts[1].strip() if len(parts) > 1 else ""

            if question_title.startswith("Does ") and summary_content:
                simplified_key = _create_simplified_key(question_title)
                if simplified_key:
                    parsed_summaries_temp[simplified_key] = f"Regarding the question: '{question_title}'\n\n{summary_content}"
                else: # Fallback if key generation fails
                    parsed_summaries_temp[f"unkeyed_summary_{i}"] = block_text
            elif not parsed_summaries_temp and i==0 : # Could be a single block summary
                 simplified_key = _create_simplified_key(question_title) if question_title.startswith("Does ") else "general_summary"
                 parsed_summaries_temp[simplified_key] = block_text


        except Exception as e:
            print(f"Minor parsing error on block {i}: {e}. Storing block as is.")
            parsed_summaries_temp[f"raw_block_{i}"] = block_text
    
    if not parsed_summaries_temp and raw_content: # If no blocks were parsed but there's content
        print("Warning: Could not parse specific causal summaries. Using full text as fallback.")
        CACHED_CAUSAL_SUMMARIES_DICT["full_summary"] = raw_content
    elif parsed_summaries_temp:
        CACHED_CAUSAL_SUMMARIES_DICT = parsed_summaries_temp
        print(f"Parsed {len(CACHED_CAUSAL_SUMMARIES_DICT)} causal summaries.")
        # print("DEBUG: Available causal summary keys:", CACHED_CAUSAL_SUMMARIES_DICT.keys())
    else:
        error_msg = "No causal summary content found or parsed."
        print(error_msg)
        CACHED_CAUSAL_SUMMARIES_DICT = {"error": error_msg}


def get_specific_causal_summary(user_keywords_list):
    """
    Tries to find a specific causal summary based on a list of user keywords.
    """
    if not CACHED_CAUSAL_SUMMARIES_DICT or "error" in CACHED_CAUSAL_SUMMARIES_DICT:
        return CACHED_CAUSAL_SUMMARIES_DICT.get("error", "Causal summary data is not available.")
    
    if "full_summary" in CACHED_CAUSAL_SUMMARIES_DICT and len(CACHED_CAUSAL_SUMMARIES_DICT) == 1:
        return CACHED_CAUSAL_SUMMARIES_DICT["full_summary"] # Only full summary available

    if not user_keywords_list: # User asked generally "causal analysis"
        # Concatenate all summaries or return a "pick one" prompt (for LLM to handle)
        all_summaries_text = "\n\n---\n\n".join(
            summary_text for key, summary_text in CACHED_CAUSAL_SUMMARIES_DICT.items() 
            if key not in ["error", "full_summary", "introduction_or_full"]
        )
        return all_summaries_text if all_summaries_text else "Several causal analyses have been performed. Please specify which one you are interested in."

    best_match_key = None
    highest_score = 0

    for key_from_kb in CACHED_CAUSAL_SUMMARIES_DICT:
        if key_from_kb in ["error", "full_summary"]:
            continue
        
        current_score = 0
        kb_key_tokens = set(key_from_kb.split())
        
        # Score based on overlap
        for user_keyword in user_keywords_list:
            if user_keyword in kb_key_tokens:
                current_score += 2 # Exact keyword match
            else: # Check for partial match for terms like "hc 4f" vs "4f"
                for kb_token in kb_key_tokens:
                    if user_keyword in kb_token or kb_token in user_keyword:
                        current_score +=1
                        break
        
        # Bonus for matching more of the specific key's terms
        if current_score > 0 :
             coverage = current_score / (len(kb_key_tokens) + len(user_keywords_list)) # Simple coverage metric
             current_score += coverage 


        if current_score > highest_score:
            highest_score = current_score
            best_match_key = key_from_kb
    
    # Define a threshold for a "good enough" match
    # This threshold might need tuning. Let's say >50% of user keywords must contribute meaningfully
    required_keyword_contribution = len(user_keywords_list) * 0.5 
    if best_match_key and highest_score > required_keyword_contribution and highest_score > 1: # At least more than one keyword or a strong single match
        print(f"DEBUG: Matched user keywords to causal summary key: '{best_match_key}' with score {highest_score}")
        return CACHED_CAUSAL_SUMMARIES_DICT[best_match_key]
    else:
        print("DEBUG: No specific causal summary matched well enough. Providing general information or full summary.")
        # Fallback to providing all summaries if a specific one isn't found
        all_summaries_text = "\n\n---\n\n".join(
            summary_text for key, summary_text in CACHED_CAUSAL_SUMMARIES_DICT.items() 
            if key not in ["error", "full_summary"]
        )
        return all_summaries_text if all_summaries_text else "Could not retrieve a specific causal summary. General causal analysis has been performed."