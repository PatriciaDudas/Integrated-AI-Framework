# evaluation_runner.py
import pandas as pd
import json
import datetime

# Import your chatbot modules
from evaluation_test_set import test_queries # Your test cases
import db_handler
import llm_handler
import knowledge_base
import pred_maint_interfacer
import analysis_module # Your analysis module with stubs or actual functions

# Import the intent recognition function (assuming it's in chatbot_app or streamlit_chatbot)
# For this example, let's assume you've refactored it into chatbot_app.py
# If it's still embedded in streamlit_chatbot.py, you might need to copy it here
# or restructure your project slightly.
from chatbot_app import get_intent_and_entities # Make sure chatbot_app.py is in the same directory or accessible

def run_evaluation():
    # Load knowledge bases once
    knowledge_base.parse_and_load_causal_summary()

    results_log = []
    
    print(f"Starting evaluation with {len(test_queries)} test queries...\n")

    for i, test_case in enumerate(test_queries):
        print(f"--- Running Test Query ID: {test_case['query_id']} ({i+1}/{len(test_queries)}) ---")
        print(f"User Query: {test_case['user_query']}")

        actual_intent, actual_entities = get_intent_and_entities(test_case['user_query'])
        print(f"Detected Intent: {actual_intent}, Detected Entities: {actual_entities}")

        context_for_llm = ""
        response_prefix = ""
        final_chatbot_response = ""
        data_fetched_for_llm = False
        llm_needed = True # Default assumption

        # This logic block is adapted from your streamlit_chatbot.py's main processing loop
        # It needs to call the same backend functions based on intent.
        if actual_intent == "get_frequent_hc_type_combos":
            context_for_llm = db_handler.get_frequent_healthcode_type_combinations(actual_entities.get("limit", 5))
            response_prefix = "Here are the most frequent healthcode/type combinations:\n"
            data_fetched_for_llm = True
            # For some intents, we might decide LLM isn't needed if context is the full answer
            llm_needed = False # Assuming direct display for this one
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_vobc_reliability":
            if "vobcid" in actual_entities:
                general_reliability_info = db_handler.get_vobcid_error_counts(actual_entities.get("limit", 5))
                specific_vobc_info = db_handler.fetch_data_for_vobcid(actual_entities["vobcid"])
                context_for_llm = f"Overall VobcId error counts:\n{general_reliability_info}\n\nDetails for VobcId {actual_entities['vobcid']}:\n{specific_vobc_info}"
            else: 
                context_for_llm = db_handler.get_vobcid_error_counts(actual_entities.get("limit", 5))
            response_prefix = "Regarding VOBC reliability:\n"
            data_fetched_for_llm = True
            llm_needed = False # Direct display
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_temporal_patterns":
            context_for_llm = db_handler.get_temporal_error_patterns(
                healthcode_filter=actual_entities.get("healthcode"),
                type_filter=actual_entities.get("type"),
                time_granularity=actual_entities.get("granularity", "hour")
            )
            response_prefix = "Regarding temporal error patterns:\n"
            data_fetched_for_llm = True
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_engineering_train_info":
            context_for_llm = db_handler.get_engineering_train_events_summary()
            response_prefix = "Regarding engineering train events:\n"
            data_fetched_for_llm = True
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)
            
        elif actual_intent == "get_excludable_errors":
            context_for_llm = db_handler.get_excludable_errors_summary()
            response_prefix = "Regarding excludable errors:\n"
            data_fetched_for_llm = True
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_location_specific_errors":
            context_for_llm = db_handler.get_location_specific_error_counts(
                location_field=actual_entities.get("location_field", "TrackSection")
            )
            response_prefix = "Regarding location-specific errors:\n"
            data_fetched_for_llm = True
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_ingestion_lag":
            context_for_llm = db_handler.get_ingestion_lag_stats()
            response_prefix = "Regarding data ingestion lag:\n"
            data_fetched_for_llm = True
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "find_sequential_errors":
            vobcid = actual_entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.find_sequential_errors(
                    vobcid=vobcid,
                    time_window_minutes=actual_entities.get("time_window_minutes", 5)
                )
            else:
                context_for_llm = "Please specify a VobcId to find sequential errors."
            data_fetched_for_llm = True
            response_prefix = "Sequential error analysis:\n" if vobcid else ""
            llm_needed = False # Assuming analysis_module returns display-ready string
            final_chatbot_response = response_prefix + str(context_for_llm)
            
        elif actual_intent == "analyze_message_params":
            hc = actual_entities.get("healthcode")
            htype = actual_entities.get("type")
            if hc:
                context_for_llm = analysis_module.analyze_message_params_for_hc_type(hc, htype)
            else:
                context_for_llm = "Please specify a Healthcode for parameter analysis."
            data_fetched_for_llm = True
            response_prefix = "Message parameter analysis:\n" if hc else ""
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "get_vccnum_correlation":
            context_for_llm = analysis_module.get_vccnum_healthcode_analysis()
            data_fetched_for_llm = True
            response_prefix = "VCC Number correlation analysis:\n"
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)

        elif actual_intent == "check_record_number_gaps":
            vobcid = actual_entities.get("vobcid")
            if vobcid:
                context_for_llm = analysis_module.check_record_number_gaps(
                    vobcid, 
                    healthcode=actual_entities.get("healthcode"), 
                    type_val=actual_entities.get("type")
                )
            else:
                context_for_llm = "Please specify a VobcId for record number gap analysis."
            data_fetched_for_llm = True
            response_prefix = "Record number gap analysis:\n" if vobcid else ""
            llm_needed = False
            final_chatbot_response = response_prefix + str(context_for_llm)
        
        elif actual_intent == "get_healthcode_description":
            hc = actual_entities.get("healthcode")
            hc_type = actual_entities.get("type")
            context_for_llm = db_handler.fetch_healthcode_description(hc, hc_type) if hc else "Could not identify healthcode."
            data_fetched_for_llm = True
            llm_needed = False # Direct answer
            final_chatbot_response = str(context_for_llm)

        elif actual_intent == "get_vobcid_data":
            vobcid = actual_entities.get("vobcid")
            context_for_llm = db_handler.fetch_data_for_vobcid(vobcid) if vobcid else "Could not identify VobcId."
            data_fetched_for_llm = True
            llm_needed = False # Direct answer
            final_chatbot_response = str(context_for_llm)

        elif actual_intent == "get_vobcid_type":
            vobcid = actual_entities.get("vobcid")
            context_for_llm = db_handler.get_current_type_for_vobcid(vobcid) if vobcid else "Could not identify VobcId."
            data_fetched_for_llm = True
            llm_needed = False # Direct answer
            final_chatbot_response = str(context_for_llm)
            
        elif actual_intent == "get_causal_summary":
            keywords = actual_entities.get("keywords_for_causal", [])
            context_for_llm = knowledge_base.get_specific_causal_summary(keywords)
            if not context_for_llm or "unavailable" in str(context_for_llm).lower() or "error" in str(context_for_llm).lower():
                final_chatbot_response = str(context_for_llm) if context_for_llm else "Causal summary not found."
                llm_needed = False
            else:
                response_prefix = "Based on our causal analysis related to your query:\n" if keywords else "Here's an overview of our causal analysis findings:\n"
                llm_needed = True # LLM will use this context
            data_fetched_for_llm = True
                 
        elif actual_intent == "get_predictive_maintenance_report":
            response_prefix = "Fetching predictive maintenance analysis...\n"
            pm_report_text = pred_maint_interfacer.get_predictive_maintenance_results()
            context_for_llm = pm_report_text
            if "Error:" in pm_report_text or "not found" in pm_report_text.lower():
                final_chatbot_response = response_prefix + pm_report_text
                llm_needed = False
            else:
                response_prefix += "Predictive maintenance report:\n"
                llm_needed = True
            data_fetched_for_llm = True
        
        elif actual_intent == "unknown":
            context_for_llm = ("User is asking a general question. Please attempt to answer based on your understanding "
                               "of rail systems and the types of data usually available in this context, "
                               "or state if it's outside the scope of typical rail operations data queries.")
            llm_needed = True 
            data_fetched_for_llm = True # Mark as fetched so LLM call happens

        # Call LLM if needed
        if llm_needed:
            final_context = str(context_for_llm) if context_for_llm is not None else ""
            # print(f"DEBUG_EVAL: Context for LLM (first 200): {final_context[:200]}")
            llm_response = llm_handler.call_mistral_model(test_case['user_query'], context_data=final_context)
            final_chatbot_response = response_prefix + llm_response
        
        print(f"Chatbot Response: {final_chatbot_response[:300]}...") # Print snippet of final response

        results_log.append({
            "query_id": test_case['query_id'],
            "user_query": test_case['user_query'],
            "expected_intent": test_case['expected_intent'],
            "actual_intent": actual_intent,
            "expected_entities": test_case.get('expected_entities', {}),
            "actual_entities": actual_entities,
            "context_provided_to_llm": str(context_for_llm)[:1000], # Log part of the context
            "final_chatbot_response": final_chatbot_response,
            "golden_answer_keywords": test_case.get('golden_answer_keywords', []),
            "is_out_of_domain": test_case['is_out_of_domain'],
            "task_category": test_case['task_category']
        })
        print("-" * 50)

    # Save results to a CSV file
    results_df = pd.DataFrame(results_log)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_results_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\nEvaluation complete. Results saved to: {output_filename}")
    
    return results_df

if __name__ == "__main__":
    evaluation_df = run_evaluation()
    # --- Basic Analysis of Results (can be expanded) ---
    print("\n--- Basic Evaluation Metrics ---")
    
    # Intent Recognition Accuracy
    correct_intents = (evaluation_df['expected_intent'] == evaluation_df['actual_intent']).sum()
    total_queries = len(evaluation_df)
    intent_accuracy = (correct_intents / total_queries) * 100 if total_queries > 0 else 0
    print(f"Intent Recognition Accuracy: {intent_accuracy:.2f}% ({correct_intents}/{total_queries})")

    # Domain Restriction (Out-of-Domain Queries)
    ood_queries = evaluation_df[evaluation_df['is_out_of_domain'] == True]
    if not ood_queries.empty:
        # Define what a "correct refusal" looks like (e.g., contains certain keywords)
        # This is a simple check, might need refinement
        refusal_keywords = ["cannot answer", "outside my scope", "don't have information", "rail domain"]
        ood_queries.loc[:, 'correctly_refused'] = ood_queries['final_chatbot_response'].apply(
            lambda r: any(kw in r.lower() for kw in refusal_keywords)
        )
        correct_refusals = ood_queries['correctly_refused'].sum()
        total_ood = len(ood_queries)
        ood_refusal_rate = (correct_refusals / total_ood) * 100 if total_ood > 0 else 0
        print(f"Out-of-Domain Refusal Rate: {ood_refusal_rate:.2f}% ({correct_refusals}/{total_ood})")
    else:
        print("No out-of-domain queries in the test set to evaluate domain restriction.")

    print("\nFurther analysis (e.g., LLM response quality, data retrieval accuracy) requires manual review or more complex keyword checking of the saved CSV.")