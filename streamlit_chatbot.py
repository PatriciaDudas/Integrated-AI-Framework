# streamlit_chatbot.py
import streamlit as st
import re # For get_intent_and_entities

# Import your existing chatbot modules
import db_handler
import llm_handler
import knowledge_base
import pred_maint_interfacer
import analysis_module # Now with stubs

# --- Initialization ---
# Load knowledge bases once at the start of the session (Streamlit caches this)
if "causal_summary_loaded" not in st.session_state:
    knowledge_base.parse_and_load_causal_summary()
    st.session_state.causal_summary_loaded = True
    
# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # Initialize as an empty list

# --- Copy your get_intent_and_entities function here directly ---
# This function is from your provided chatbot_app.py
def get_intent_and_entities(user_query):
    query_lower = user_query.lower()
    intent = "unknown"
    entities = {"keywords_for_causal": []} 

    # Order matters: more specific regex first
    # Q0: Error Frequency
    if re.search(r"(most frequent|top)\s*(healthcode|error|event|error code)s?\s", query_lower):
        intent = "get_frequent_hc"
        limit_match = re.search(r"top\s*(\d+)", query_lower)
        entities["limit"] = int(limit_match.group(1)) if limit_match else 5
        return intent, entities
    
    # Q1: Error Frequency and Description
    if re.search(r"(most frequent|top)\s*(healthcode|error|event)s?\s*(and type|type and)?\s*(combination|combo|pattern| )", query_lower):
        intent = "get_frequent_hc_type_combos"
        limit_match = re.search(r"top\s*(\d+)", query_lower)
        entities["limit"] = int(limit_match.group(1)) if limit_match else 5
        return intent, entities

    # Q3: Temporal Error Patterns
    if re.search(r"(temporal|time|when).*(?:pattern|occur|happen)?.*(error|healthcode)", query_lower):
        intent = "get_temporal_patterns"
        if "hour" in query_lower: entities["granularity"] = "hour"
        elif "day of week" in query_lower or "days of the week" in query_lower : entities["granularity"] = "day_of_week"
        elif "date" in query_lower: entities["granularity"] = "date"
        else: entities["granularity"] = "hour" # Default
        
        hc_match = re.search(r"(?:for|of)\s*healthcode\s*([0-9A-Fa-f]{1,2})", query_lower)
        if hc_match: entities["healthcode"] = hc_match.group(1).upper()
        type_match = re.search(r"type\s*(\d+)", query_lower)
        if type_match: entities["type"] = int(type_match.group(1))
        return intent, entities

    # Q4: Impact of Engineering Trains
    if re.search(r"(engineering train|engineer work|EngineeringTrainOnly)", user_query, re.IGNORECASE): # Case insensitive for this
        intent = "get_engineering_train_info"
        return intent, entities
        
    # Q5: Excludable Errors
    if re.search(r"(excludable|exclude = 1|non-critical)\s*(error|healthcode)", query_lower):
        intent = "get_excludable_errors"
        return intent, entities

    # Q7: Location-Specific Errors
    loc_match = re.search(r"(location|tracksection|segmentid|loopnum|xovernum).*(?:specific|frequent)?.*(error|healthcode)", query_lower)
    if loc_match:
        intent = "get_location_specific_errors"
        if "tracksection" in query_lower: entities["location_field"] = "TrackSection"
        elif "segmentid" in query_lower: entities["location_field"] = "SegmentId"
        elif "loopnum" in query_lower: entities["location_field"] = "LoopNum"
        elif "xovernum" in query_lower: entities["location_field"] = "XOverNum"
        else: entities["location_field"] = "TrackSection" # Default
        return intent, entities
        
    # Q8: Data Ingestion Lag
    if re.search(r"(ingestion lag|delay between.*timestamp and ingestiondate|timestamp vs ingestion|timestamp and ingestion|timestamp)", query_lower):
        intent = "get_ingestion_lag"
        return intent, entities

    # Q9: Sequential Errors
    seq_match = re.search(r"(sequential|sequence of|precede|follow).*(error|healthcode)", query_lower)
    if seq_match:
        intent = "find_sequential_errors"
        vobcid_m = re.search(r"(?:for|on|of)\s*vobcid\s*(\d+)", query_lower)
        if vobcid_m: entities["vobcid"] = int(vobcid_m.group(1))
        entities["time_window_minutes"] = 5 
        time_m = re.search(r"within\s*(\d+)\s*minute", query_lower)
        if time_m: entities["time_window_minutes"] = int(time_m.group(1))
        return intent, entities

    # Q10: Parameter Analysis
    param_match = re.search(r"analyze\s*(parameter|param1|param2|param3)s?.*(?:for|of)\s*(healthcode|hc)\s*([0-9A-Fa-f]{1,2})", query_lower)
    if param_match:
        intent = "analyze_message_params"
        entities["healthcode"] = param_match.group(4).upper()
        type_m = re.search(r"type\s*(\d+)", query_lower)
        if type_m: entities["type"] = int(type_m.group(1))
        return intent, entities

    # Q11: VCC Number Correlation
    if re.search(r"(vccnum|vcc number).*(correlation|relation|pattern).*(healthcode|error|type)", query_lower):
        intent = "get_vccnum_correlation"
        return intent, entities

    # Q12: Record Number Gaps/Resets
    rec_num_match = re.search(r"(record number|record_num).*(gap|reset|sequence|continuity)", query_lower)
    if rec_num_match:
        intent = "check_record_number_gaps"
        vobcid_m = re.search(r"(?:for|on|of)\s*vobcid\s*(\d+)", query_lower)
        if vobcid_m: entities["vobcid"] = int(vobcid_m.group(1))
        hc_m = re.search(r"healthcode\s*([0-9A-Fa-f]{1,2})", query_lower)
        if hc_m: entities["healthcode"] = hc_m.group(1).upper()
        type_m = re.search(r"type\s*(\d+)", query_lower)
        if type_m: entities["type"] = int(type_m.group(1))
        return intent, entities

    # Q2: VOBC Reliability
    if re.search(r"(vobcid|train|vehicle).*(reliability|most error|error prone|disproportionate report)", query_lower) or \
       (intent == "unknown" and re.search(r"vobcid\s*(\d+).*(error|healthcode)", query_lower)):
        intent = "get_vobc_reliability"
        vobcid_m = re.search(r"(vobcid|train|vehicle id|id)\s*(\d+)", query_lower)
        if vobcid_m: entities["vobcid"] = int(vobcid_m.group(2))
        limit_match = re.search(r"top\s*(\d+)", query_lower)
        entities["limit"] = int(limit_match.group(1)) if limit_match else 5
        return intent, entities
        
    # Look for the core entities 'healthcode [code]' and optionally 'type [num]'
    hc_match = re.search(r"healthcode\s*([0-9A-Fa-f]{1,2})", query_lower)
    type_match = re.search(r"type\s*(\d+)", query_lower) # Keep this simple search

    # Keywords indicating a description request
    description_keywords = ["description", "meaning", "define", "details", "what is", "what's", "tell me about", "info on", "info for", "show me", "show", "details for", "status of"]

    # Check if healthcode is present AND at least one description keyword exists
    if hc_match and any(keyword in query_lower for keyword in description_keywords):
        # Ensure this intent isn't better handled by a more specific analysis query first
        # (This depends on the order of checks in your function. Assume this comes after more specific Q-intents)

        intent = "get_healthcode_description"
        entities["healthcode"] = hc_match.group(1).upper()
        if type_match:
            entities["type"] = int(type_match.group(1))
        # Potentially add logic here to check if a Type was expected but not found, if necessary

        return intent, entities

    vobcid_match_orig = re.search(r"^(what is the|type for|data for|info on|details for|status of|show.*events for) (?:train|vobcid|vehicle id|id)\s*(\d+)", query_lower) # Added ^
    if vobcid_match_orig:
        entities["vobcid"] = int(vobcid_match_orig.group(2))
        if "type for" in query_lower: intent = "get_vobcid_type"
        else: intent = "get_vobcid_data"
        return intent, entities

    # --- Broader Causal Analysis Intent Detection (from previous version) ---
    causal_topic_keywords = ["slip slide", "Slip/Slide", "door failure", "simulated transponder", "hc 4f", "hc 45", "hc f9", "hc 7b", "type 0", "type 2", "type 8", "ts 50345", "ts 20467"]
    causal_structure_indicators = [r"does .* cause", r"what is the cause of", r"effect of .* on", r"impact of .* on", r"does .* lead to", "lead to", r"what does .* mean"]
    general_causal_terms = ["causal analysis", "causal summary", "causal findings", "causality"]

    matched_causal_topic = any(topic_kw in query_lower for topic_kw in causal_topic_keywords)
    matched_causal_structure = any(re.search(pattern, query_lower) for pattern in causal_structure_indicators)
    matched_general_causal = any(term in query_lower for term in general_causal_terms)

    if matched_causal_structure or matched_general_causal or (("tell me about" in query_lower or "what about" in query_lower) and matched_causal_topic):
        intent = "get_causal_summary"
        norm_query = query_lower
        norm_query = re.sub(r"what is|what's|what did|tell me about|what can you tell me about|does|is there|an|the|of|for|on|vs|cause|effect|impact|probability|likelihood|related to|operational mode|what is the definition|define", "", norm_query)
        norm_query = re.sub(r"health codes|healthcode", "hc", norm_query) 
        norm_query = re.sub(r"track sections|tracksection|section", "ts", norm_query)
        norm_query = re.sub(r"type\s+", "type ", norm_query)
        norm_query = re.sub(r"[,\.\?'\(\)]", "", norm_query)
        extracted_tokens = norm_query.split()
        meaningful_keywords = []
        temp_tokens = list(filter(None, extracted_tokens))
        i = 0
        while i < len(temp_tokens):
            token = temp_tokens[i].strip()
            if not token: i += 1; continue
            if token == "slip" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip() == "slide":
                meaningful_keywords.append("slip slide"); i += 1
            elif token == "door" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip() == "failures":
                meaningful_keywords.append("door failures"); i += 1
            elif token == "simulated" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip() == "transponders":
                meaningful_keywords.append("simulated transponders"); i += 1
            elif token == "type" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip().isalnum():
                meaningful_keywords.append(f"type {temp_tokens[i+1].strip()}"); i += 1
            elif token == "hc" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip().isalnum() and len(temp_tokens[i+1].strip()) <=2 :
                meaningful_keywords.append(f"hc {temp_tokens[i+1].strip()}"); i += 1
            elif token == "ts" and i + 1 < len(temp_tokens) and temp_tokens[i+1].strip().isdigit():
                meaningful_keywords.append(f"ts {temp_tokens[i+1].strip()}"); i += 1
            elif token.isalnum() and not token.isdigit():
                if token not in ["type", "hc", "ts"]:
                    meaningful_keywords.append(token)
            i += 1
        entities["keywords_for_causal"] = sorted(list(set(filter(None, meaningful_keywords))))
        return intent, entities

    # Ensure pred_maint intent is checked after more specific ones
    pred_maint_keywords_orig = ["predictive maintenance", "maintenance prediction", "rul", "anomaly detection", "needs maintenance", "predictive analysis", "predictive report", "predictive maintenance report", "predictive maintenance analysis", "predictive maintenance results"]
    if any(keyword in query_lower for keyword in pred_maint_keywords_orig):
        intent = "get_predictive_maintenance_report"
        vobcid_pm_match = re.search(r"(?:for|on|about) (?:train|vobcid|vehicle id|id)\s*(\d+)", query_lower)
        if vobcid_pm_match: entities["vobcid"] = int(vobcid_pm_match.group(1))
        return intent, entities
        
    return intent, entities
# --- End of get_intent_and_entities ---

# --- Display existing messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# --- Main Chat Logic (Revised) ---
if user_input := st.chat_input("Ask about rail operations..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Used for streaming-like effect if LLM supports it
        message_placeholder.markdown("...") 
        
        intent, entities = get_intent_and_entities(user_input)
        # st.write(f"DEBUG: Intent={intent}, Entities={entities}") # Optional: display debug info in UI

        # Initialize variables for this turn
        response_text = ""        # This will hold the final text displayed
        context_for_llm = ""      # This will hold data fetched for the LLM
        response_prefix = ""      # Optional text to add before the LLM response
        error_occurred = False    # Flag if context fetching failed

        # --- Step 1: Fetch context based on intent ---
        # (Mirroring the logic from chatbot_app.py main loop)
        if intent == "get_frequent_hc":
            context_for_llm = db_handler.get_frequent_healthcode(entities.get("limit", 5))
            response_prefix = "Here are the most frequent healthcode errors found:\n"
        elif intent == "get_frequent_hc_type_combos":
            context_for_llm = db_handler.get_frequent_healthcode_type_combinations(entities.get("limit", 5))
            response_prefix = "Here are the most frequent healthcode/type combinations found:\n"
        elif intent == "get_vobc_reliability":
            if "vobcid" in entities:
                general_reliability_info = db_handler.get_vobcid_error_counts(entities.get("limit", 5))
                specific_vobc_info = db_handler.fetch_data_for_vobcid(entities["vobcid"])
                context_for_llm = f"Overall VobcId error counts:\n{general_reliability_info}\n\nDetails for VobcId {entities['vobcid']}:\n{specific_vobc_info}"
                response_prefix = f"Regarding VOBC reliability, especially for {entities['vobcid']}:\n"
            else:
                context_for_llm = db_handler.get_vobcid_error_counts(entities.get("limit", 5))
                response_prefix = "Here's an overview of VOBC reliability based on error counts:\n"
        elif intent == "get_temporal_patterns":
            context_for_llm = db_handler.get_temporal_error_patterns(
                healthcode_filter=entities.get("healthcode"),
                type_filter=entities.get("type"),
                time_granularity=entities.get("granularity", "hour")
                # Consider adding limit here if needed: limit=entities.get("limit", 10)
            )
            response_prefix = f"Regarding temporal patterns (granularity: {entities.get('granularity', 'hour')}):\n"
        elif intent == "get_engineering_train_info":
            context_for_llm = db_handler.get_engineering_train_events_summary()
            response_prefix = "Summary of events potentially related to engineering trains:\n"
        elif intent == "get_excludable_errors":
            context_for_llm = db_handler.get_excludable_errors_summary()
            response_prefix = "Summary regarding excludable errors:\n"
        elif intent == "get_location_specific_errors":
            context_for_llm = db_handler.get_location_specific_error_counts(
                location_field=entities.get("location_field", "TrackSection")
                # Add limits here if needed
            )
            response_prefix = f"Regarding location-specific errors (by {entities.get('location_field', 'TrackSection')}):\n"
        elif intent == "get_ingestion_lag":
            context_for_llm = db_handler.get_ingestion_lag_stats()
            response_prefix = "Analysis of data ingestion lag:\n"
        elif intent == "find_sequential_errors":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.find_sequential_errors(
                    vobcid=vobcid,
                    time_window_minutes=entities.get("time_window_minutes", 5)
                )
                response_prefix = f"Sequential error analysis for VobcId {vobcid}:\n"
            else:
                response_text = "Please specify a VobcId to find sequential errors."
                error_occurred = True # Set final response directly, skip LLM
        elif intent == "analyze_message_params":
            hc = entities.get("healthcode")
            htype = entities.get("type")
            if hc:
                context_for_llm = analysis_module.analyze_message_params_for_hc_type(hc, htype)
                type_str = f" Type {htype}" if htype is not None else ""
                response_prefix = f"Parameter analysis for Healthcode {hc}{type_str}:\n"
            else:
                response_text = "Please specify a Healthcode for parameter analysis."
                error_occurred = True # Set final response directly, skip LLM
        elif intent == "get_vccnum_correlation":
            context_for_llm = analysis_module.get_vccnum_healthcode_analysis()
            response_prefix = "Analysis of VccNum and Healthcode correlation:\n"
        elif intent == "check_record_number_gaps":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.check_record_number_gaps(
                    vobcid,
                    healthcode=entities.get("healthcode"),
                    type_val=entities.get("type")
                )
                response_prefix = f"Record number gap analysis for VobcId {vobcid}:\n"
            else:
                response_text = "Please specify a VobcId for record number gap analysis."
                error_occurred = True # Set final response directly, skip LLM
        elif intent == "get_healthcode_description":
            hc = entities.get("healthcode")
            hc_type = entities.get("type")
            if hc:
                context_for_llm = db_handler.fetch_healthcode_description(hc, hc_type)
                response_prefix = f"Description for Healthcode {hc}"
                if hc_type is not None: response_prefix += f" Type {hc_type}"
                response_prefix += ":\n"
            else:
                response_text = "Could not identify the healthcode in your question."
                error_occurred = True
        elif intent == "get_vobcid_data":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = db_handler.fetch_data_for_vobcid(vobcid)
                response_prefix = f"Data summary for VobcId {vobcid}:\n"
            else:
                response_text = "Could not identify the VobcId in your question."
                error_occurred = True
        elif intent == "get_vobcid_type":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = db_handler.get_current_type_for_vobcid(vobcid)
                response_prefix = f"Regarding Type for VobcId {vobcid}:\n"
            else:
                response_text = "Could not identify the VobcId in your question."
                error_occurred = True
        elif intent == "get_causal_summary":
            keywords = entities.get("keywords_for_causal", [])
            context_for_llm = knowledge_base.get_specific_causal_summary(keywords)
            # Check if fetching context itself resulted in an error/unavailable message
            if not context_for_llm or "unavailable" in str(context_for_llm).lower() or "error" in str(context_for_llm).lower():
                response_text = str(context_for_llm) if context_for_llm else "Sorry, I couldn't retrieve specific causal analysis information."
                error_occurred = True # Set final response directly, skip LLM
                context_for_llm = "" # Clear context so LLM isn't called
            else:
                 response_prefix = "Based on our causal analysis related to your query:\n" if keywords else "Here's an overview of our causal analysis findings:\n"
        elif intent == "get_predictive_maintenance_report":
            response_prefix = "Fetching predictive maintenance analysis...\n"
            pm_report_text = pred_maint_interfacer.get_predictive_maintenance_results()
            context_for_llm = pm_report_text
            # Check if fetching context resulted in an error
            if "Error:" in pm_report_text or "not found" in pm_report_text.lower():
                response_text = response_prefix + pm_report_text # Show the error directly
                error_occurred = True # Set final response directly, skip LLM
                context_for_llm = "" # Clear context
            else:
                response_prefix += "Predictive maintenance report:\n"
        elif intent == "unknown":
            context_for_llm = ("User is asking a general question. Please attempt to answer based on your understanding "
                               "of rail systems and the types of data usually available in this context, "
                               "or state if it's outside the scope of typical rail operations data queries.")
            # No prefix needed, LLM will handle entirely

        # --- Step 2: Call LLM if no error occurred during context fetching ---
        # LLM is needed if we have context, OR if the intent was 'unknown'
        if not error_occurred and (context_for_llm or intent == "unknown"):
            # Ensure context_for_llm is a string
            final_context = str(context_for_llm) if context_for_llm is not None else ""

            # Optional: Debug log for context being sent
            # st.write(f"DEBUG: Context for LLM (first 500 chars): \n'{final_context[:500]}...' \n" if final_context else "DEBUG: No specific context for LLM (general query).")

            llm_response = llm_handler.call_mistral_model(user_input, context_data=final_context)
            response_text = response_prefix + llm_response # Combine prefix and LLM response
        elif not error_occurred and response_prefix:
             # Handle cases where only a prefix was set but no context/LLM call needed (should be rare with this logic)
             response_text = response_prefix
        elif not error_occurred:
             # Fallback if no error, no context, no prefix, and not unknown intent
             response_text = "I understood the intent but couldn't retrieve relevant information or generate a response. Please try rephrasing."

        # --- Step 3: Display the final response ---
        message_placeholder.markdown(response_text) # response_text now contains the final answer
        st.session_state.messages.append({"role": "assistant", "content": response_text})