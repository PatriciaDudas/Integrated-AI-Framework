# chatbot_app.py
import re
import db_handler
import llm_handler
import knowledge_base
import pred_maint_interfacer
import analysis_module # New import

def get_intent_and_entities(user_query):
    query_lower = user_query.lower()
    intent = "unknown"
    entities = {"keywords_for_causal": []} 

    # Order matters: more specific regex first
    
    # Q1: Error Frequency and Description
    if re.search(r"(most frequent|top)\s*(healthcode|error|event)s?\s*(and type|type and)?\s*(combination|combo|pattern)", query_lower):
        intent = "get_frequent_hc_type_combos"
        limit_match = re.search(r"top\s*(\d+)", query_lower)
        entities["limit"] = int(limit_match.group(1)) if limit_match else 5
        return intent, entities

    # Q3: Temporal Error Patterns
    if re.search(r"(temporal|time|when)\s*(pattern|occur|happen).*(error|healthcode)", query_lower):
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
    loc_match = re.search(r"(location|tracksection|segmentid|loopnum|xovernum).*(specific|frequent).*(error|healthcode)", query_lower)
    if loc_match:
        intent = "get_location_specific_errors"
        if "tracksection" in query_lower: entities["location_field"] = "TrackSection"
        elif "segmentid" in query_lower: entities["location_field"] = "SegmentId"
        elif "loopnum" in query_lower: entities["location_field"] = "LoopNum"
        elif "xovernum" in query_lower: entities["location_field"] = "XOverNum"
        else: entities["location_field"] = "TrackSection" # Default
        return intent, entities
        
    # Q8: Data Ingestion Lag
    if re.search(r"(ingestion lag|delay between.*timestamp and ingestiondate|timestamp vs ingestion)", query_lower):
        intent = "get_ingestion_lag"
        return intent, entities

    # Q9: Sequential Errors
    seq_match = re.search(r"(sequential|sequence of|precede|follow).*(error|healthcode)", query_lower)
    if seq_match:
        intent = "find_sequential_errors"
        vobcid_m = re.search(r"(?:for|on|of)\s*vobcid\s*(\d+)", query_lower)
        if vobcid_m: entities["vobcid"] = int(vobcid_m.group(1))
        # Default time window if not specified
        entities["time_window_minutes"] = 5 
        time_m = re.search(r"within\s*(\d+)\s*minute", query_lower)
        if time_m: entities["time_window_minutes"] = int(time_m.group(1))
        return intent, entities

    # Q10: Parameter Analysis
    param_match = re.search(r"(parameter|param1|param2|param3).*(analysis|value|pattern).*(healthcode|hc)\s*([0-9A-Fa-f]{1,2})", query_lower)
    if param_match:
        intent = "analyze_message_params"
        entities["healthcode"] = param_match.group(4).upper()
        type_m = re.search(r"type\s*(\d+)", query_lower)
        if type_m: entities["type"] = int(type_m.group(1))
        else: # Try to get type from HC_Descriptions if only HC is specified (more complex, might need to be done in main)
            print("Parameter analysis: Type not specified, will attempt for default or all types if HC_Description allows.")
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

    # Q2: VOBC Reliability (can be triggered by VobcId questions too)
    if re.search(r"(vobcid|train|vehicle).*(reliability|most error|error prone|disproportionate report)", query_lower) or \
       (intent == "unknown" and re.search(r"vobcid\s*(\d+).*(error|healthcode)", query_lower)): # Catch specific vobcid error queries
        intent = "get_vobc_reliability"
        vobcid_m = re.search(r"(vobcid|train|vehicle id|id)\s*(\d+)", query_lower)
        if vobcid_m: entities["vobcid"] = int(vobcid_m.group(2)) # Check if for a specific VOBC
        limit_match = re.search(r"top\s*(\d+)", query_lower)
        entities["limit"] = int(limit_match.group(1)) if limit_match else 5
        return intent, entities
        
    # --- Keep existing intents from previous implementation ---
    hc_match = re.search(r"(what is|define|description of|tell me about) healthcode\s*([0-9A-Fa-f]{1,2})", query_lower)
    if hc_match:
        intent = "get_healthcode_description"
        entities["healthcode"] = hc_match.group(2).upper()
        type_match = re.search(r"type\s*(\d+)", query_lower)
        if type_match: entities["type"] = int(type_match.group(1))
        return intent, entities

    vobcid_match = re.search(r"(what is the|type for|data for|info on|details for|status of) (?:train|vobcid|vehicle id|id)\s*(\d+)", query_lower)
    if vobcid_match:
        entities["vobcid"] = int(vobcid_match.group(2))
        if "type for" in query_lower: intent = "get_vobcid_type"
        else: intent = "get_vobcid_data"
        return intent, entities
     # --- Broader Causal Analysis Intent Detection ---
    # Keywords that strongly suggest a specific causal question topic
    causal_topic_keywords = [
        "slip slide", "door failure", "simulated transponder", "healthcode 4f", "hc 4f",
        "healthcode 45", "hc 45", "healthcode f9", "hc f9", "healthcode 7b", "hc 7b",
        "type 0", "type 2", "type 8", 
        "tracksection 50345", "ts 50345", "tracksection 20467", "ts 20467"
        # Add more specific entity combinations from your causal summary keys if needed
    ]
    # Indicators of a causal question structure
    causal_structure_indicators = [
        r"does .* cause", r"what is the cause of", r"effect of .* on", 
        r"impact of .* on", r"why does .* lead to", "lead to"
    ]
    # General terms asking about causal analysis
    general_causal_terms = ["causal analysis", "causal summary", "causal findings", "causality"]

    matched_causal_topic = any(topic_kw in query_lower for topic_kw in causal_topic_keywords)
    matched_causal_structure = any(re.search(pattern, query_lower) for pattern in causal_structure_indicators)
    matched_general_causal = any(term in query_lower for term in general_causal_terms)

    # If any causal indicator is present, or if the query seems to be about a known causal topic
    if matched_causal_structure or matched_general_causal or (("tell me about" in query_lower or "what about" in query_lower) and matched_causal_topic):
        intent = "get_causal_summary"
        
        # Keyword extraction for causal questions (even if general, to help LLM focus if it gets full summary)
        norm_query = query_lower
        norm_query = re.sub(r"what is|what's|what did|tell me about|does|is there|an|the|of|for|on|vs|cause|effect|impact|probability|likelihood|related to|operational mode", "", norm_query)
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
            elif token.isalnum() and not token.isdigit(): # General alphanumeric terms
                if token not in ["type", "hc", "ts"]: # Avoid adding the trigger words themselves if not part of entity
                    meaningful_keywords.append(token)
            i += 1
        
        entities["keywords_for_causal"] = sorted(list(set(filter(None, meaningful_keywords))))
        return intent, entities

    pred_maint_keywords = ["predictive maintenance", "maintenance prediction", "rul", "anomaly detection", "needs maintenance"]
    if any(keyword in query_lower for keyword in pred_maint_keywords):
        intent = "get_predictive_maintenance_report" # Changed from run_...
        vobcid_pm_match = re.search(r"(?:for|on|about) (?:train|vobcid|vehicle id|id)\s*(\d+)", query_lower)
        if vobcid_pm_match: entities["vobcid"] = int(vobcid_pm_match.group(1)) # LLM can decide if it uses this
        return intent, entities
        
    return intent, entities


def main():
    print("Rail Operations Chatbot Initialized. Type 'quit' to exit.")
    # knowledge_base.load_causal_summary() # Load once
    knowledge_base.parse_and_load_causal_summary()

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'quit': break

        intent, entities = get_intent_and_entities(user_query)
        print(f"DEBUG: Intent={intent}, Entities={entities}") # For debugging

        context_for_llm = ""
        response_prefix = ""
        data_fetched = False # Flag to track if data was fetched

        # Handle specific data fetching/analysis based on intent
        if intent == "get_frequent_hc_type_combos":
            context_for_llm = db_handler.get_frequent_healthcode_type_combinations(entities.get("limit", 5))
            data_fetched = True
        elif intent == "get_vobc_reliability":
            if "vobcid" in entities: # Check info for a specific VOBC
                # This might need a new function or more complex logic
                # For now, let's provide general info and its specific data
                general_reliability_info = db_handler.get_vobcid_error_counts(entities.get("limit", 5))
                specific_vobc_info = db_handler.fetch_data_for_vobcid(entities["vobcid"])
                context_for_llm = f"Overall VobcId error counts:\n{general_reliability_info}\n\nDetails for VobcId {entities['vobcid']}:\n{specific_vobc_info}"
            else: # General question about which VobcId reports most errors
                context_for_llm = db_handler.get_vobcid_error_counts(entities.get("limit", 5))
            data_fetched = True
        elif intent == "get_temporal_patterns":
            context_for_llm = db_handler.get_temporal_error_patterns(
                healthcode_filter=entities.get("healthcode"),
                type_filter=entities.get("type"),
                time_granularity=entities.get("granularity", "hour"),
                limit=entities.get("limit", 10)
            )
            data_fetched = True
        elif intent == "get_engineering_train_info":
            context_for_llm = db_handler.get_engineering_train_events_summary()
            data_fetched = True
        elif intent == "get_excludable_errors":
            context_for_llm = db_handler.get_excludable_errors_summary()
            data_fetched = True
        elif intent == "get_location_specific_errors":
            context_for_llm = db_handler.get_location_specific_error_counts(
                location_field=entities.get("location_field", "TrackSection"),
                limit_per_location=entities.get("limit_per_location", 3),
                top_locations=entities.get("top_locations", 3)
            )
            data_fetched = True
        elif intent == "get_ingestion_lag":
            context_for_llm = db_handler.get_ingestion_lag_stats()
            data_fetched = True
        elif intent == "find_sequential_errors":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.find_sequential_errors(
                    vobcid=vobcid,
                    time_window_minutes=entities.get("time_window_minutes", 5)
                )
            else:
                context_for_llm = "Please specify a VobcId to find sequential errors."
            data_fetched = True
        elif intent == "analyze_message_params":
            hc = entities.get("healthcode")
            htype = entities.get("type") # This might be None
            if hc:
                if htype is None: # If type not specified, maybe we analyze for the most common type of this HC
                    # This part is more complex - for now, require type or assume type 0 if common
                    print("Parameter analysis: Type not explicitly specified. Provide type for more specific analysis or the bot will make an assumption.")
                    # Let LLM handle the ambiguity or fetch common type for HC
                context_for_llm = analysis_module.analyze_message_params_for_hc_type(hc, htype) # htype can be None
            else:
                context_for_llm = "Please specify a Healthcode for parameter analysis."
            data_fetched = True
        elif intent == "get_vccnum_correlation":
            context_for_llm = analysis_module.get_vccnum_healthcode_analysis() # Using analysis_module as it might need pandas
            data_fetched = True
        elif intent == "check_record_number_gaps":
            vobcid = entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.check_record_number_gaps(
                    vobcid, 
                    healthcode=entities.get("healthcode"), 
                    type_val=entities.get("type")
                )
            else:
                context_for_llm = "Please specify a VobcId for record number gap analysis."
            data_fetched = True
        
        # --- Existing intents ---
        elif intent == "get_healthcode_description":
            hc = entities.get("healthcode")
            hc_type = entities.get("type")
            if hc: context_for_llm = db_handler.fetch_healthcode_description(hc, hc_type)
            else: context_for_llm = "Could not identify the healthcode in your question."
            data_fetched = True
        elif intent == "get_vobcid_data":
            vobcid = entities.get("vobcid")
            if vobcid: context_for_llm = db_handler.fetch_data_for_vobcid(vobcid)
            else: context_for_llm = "Could not identify the VobcId in your question."
            data_fetched = True
        elif intent == "get_vobcid_type":
            vobcid = entities.get("vobcid")
            if vobcid: context_for_llm = db_handler.get_current_type_for_vobcid(vobcid)
            else: context_for_llm = "Could not identify the VobcId in your question."
            data_fetched = True
            
        elif intent == "get_causal_summary":
            keywords = entities.get("keywords_for_causal", [])
            print(f"DEBUG: Keywords for causal summary matching: {keywords}")
            context_for_llm = knowledge_base.get_specific_causal_summary(keywords)
            
            if not context_for_llm or "unavailable" in context_for_llm.lower() or "error" in context_for_llm.lower():
                response_prefix = context_for_llm if context_for_llm else "Sorry, I couldn't retrieve specific causal analysis information."
                context_for_llm = "" 
                # For errors in knowledge base, print directly and skip LLM
                print(f"Chatbot: {response_prefix}")
                continue # Go to next iteration of while loop
            elif not keywords : 
                 response_prefix = "Here's a general overview of our causal analysis findings:\n"
            else: 
                 response_prefix = "Based on our causal analysis related to your query:\n"

                 
        elif intent == "get_predictive_maintenance_report":
            response_prefix = "Fetching predictive maintenance analysis...\n"
            pm_report_text = pred_maint_interfacer.get_predictive_maintenance_results()
            context_for_llm = pm_report_text
            if "Error:" in pm_report_text or "not found" in pm_report_text.lower() :
                response_prefix += pm_report_text 
                context_for_llm = "" 
            else:
                response_prefix += "Predictive maintenance report:\n"
            data_fetched = True
        else: # intent == "unknown"
            context_for_llm = ("User is asking a general question. Please attempt to answer based on your understanding "
                               "of rail systems and the types of data usually available in this context, "
                               "or state if it's outside the scope of typical rail operations data queries.")
            data_fetched = True # We'll send it to the LLM anyway

        # Call the LLM
        if data_fetched and (context_for_llm or intent == "unknown"):
            print(f"DEBUG: Context for LLM (first 500 chars if long): \n'{str(context_for_llm)[:500]}...' \n" if context_for_llm else "DEBUG: No specific context pre-fetched for LLM (general query).")
            llm_response = llm_handler.call_mistral_model(user_query, context_data=str(context_for_llm)) # Ensure context is string
            print(f"Chatbot: {response_prefix}{llm_response}")
        elif response_prefix: 
             print(f"Chatbot: {response_prefix}") # For cases where local logic provides full response
        else:
             print("Chatbot: I'm not sure how to respond to that. Could you try rephrasing your question or check the available functionalities?")


if __name__ == "__main__":
    main()