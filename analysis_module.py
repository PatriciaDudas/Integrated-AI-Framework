# analysis_module.py
import pandas as pd
import re
from db_handler import get_db_connection # To fetch data if needed

def parse_message_params(message_str):
    """Parses Param1, Param2, Param3, and record_num from a message string."""
    params = {"Param1": None, "Param2": None, "Param3": None, "record_num": None}
    
    param_matches = re.findall(r"Param(\d):\s*(0x[0-9a-fA-F]+|\d+)", message_str)
    for num, val in param_matches:
        params[f"Param{num}"] = val
        
    record_match = re.search(r"record_num:\s*(\d+)", message_str)
    if record_match:
        params["record_num"] = int(record_match.group(1))
        
    return params

def analyze_message_params_for_hc_type(healthcode, type_val, limit=100):
    """Fetches messages for a specific HC/Type and analyzes parameter patterns."""
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    
    messages_data = []
    try:
        with conn.cursor() as cur:
            query = """
                SELECT Message FROM rail_data 
                WHERE Healthcode = %s AND Type = %s 
                LIMIT %s;
            """
            cur.execute(query, (str(healthcode), int(type_val), limit))
            messages_data = [row[0] for row in cur.fetchall()]
    except psycopg2.Error as e:
        return f"Error fetching messages for HC {healthcode} Type {type_val}: {e}"
    finally:
        if conn: conn.close()

    if not messages_data:
        return f"No messages found for Healthcode {healthcode} and Type {type_val} (sample limit {limit})."

    parsed_params_list = [parse_message_params(msg) for msg in messages_data]
    df_params = pd.DataFrame(parsed_params_list)
    
    summary = f"Parameter Analysis for HC {healthcode}, Type {type_val} (sample of {len(messages_data)} messages):\n"
    for col in ["Param1", "Param2", "Param3"]:
        if col in df_params and df_params[col].notna().any():
            summary += f"\nTop 5 common values for {col}:\n{df_params[col].value_counts().nlargest(5).to_string()}\n"
        else:
            summary += f"\nNo data or varying data for {col}.\n"
    return summary

def find_sequential_errors(vobcid, time_window_minutes=5, max_results=10):
    """
    Finds sequences of 2 healthcodes for a VobcId within a time window.
    This is a simplified version. True sequential pattern mining is more complex.
    """
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    
    try:
        with conn.cursor() as cur:
            # Fetch relevant data, ordered by time
            query = """
                SELECT Timestamp, Healthcode, Type 
                FROM rail_data 
                WHERE VobcId = %s
                ORDER BY Timestamp;
            """
            cur.execute(query, (vobcid,))
            results = cur.fetchall()
            if not results:
                return f"No data found for VobcId {vobcid} to analyze sequences."

            df_vobc = pd.DataFrame(results, columns=["Timestamp", "Healthcode", "Type"])
            if len(df_vobc) < 2:
                return f"Not enough events for VobcId {vobcid} to find sequences."

            df_vobc["Prev_HC"] = df_vobc["Healthcode"].shift(1)
            df_vobc["Prev_Timestamp"] = df_vobc["Timestamp"].shift(1)
            
            # Calculate time difference
            df_vobc["Time_Diff_Seconds"] = (df_vobc["Timestamp"] - df_vobc["Prev_Timestamp"]).dt.total_seconds()
            
            # Filter for sequences within the time window
            sequences = df_vobc[
                (df_vobc["Time_Diff_Seconds"] > 0) & 
                (df_vobc["Time_Diff_Seconds"] <= time_window_minutes * 60)
            ]
            
            if sequences.empty:
                return f"No sequential errors found within {time_window_minutes} minutes for VobcId {vobcid}."

            # Count frequent sequences
            frequent_sequences = sequences.groupby(["Prev_HC", "Healthcode"]).size().reset_index(name="count")
            frequent_sequences = frequent_sequences.sort_values("count", ascending=False).head(max_results)
            
            return f"Most frequent error sequences (HC1 -> HC2) for VobcId {vobcid} within {time_window_minutes} mins:\n{frequent_sequences.to_string(index=False)}"

    except psycopg2.Error as e:
        return f"Error fetching data for sequential analysis: {e}"
    except Exception as e_pd: # Catch pandas errors too
         return f"Error processing data for sequential analysis: {e_pd}"
    finally:
        if conn: conn.close()

def check_record_number_gaps(vobcid, healthcode=None, type_val=None, sample_limit=1000):
    """Checks for gaps or resets in record_num parsed from Message."""
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    
    messages_data = []
    try:
        with conn.cursor() as cur:
            filters = ["VobcId = %s"]
            params = [vobcid]
            if healthcode:
                filters.append("Healthcode = %s")
                params.append(healthcode)
            if type_val is not None:
                filters.append("Type = %s")
                params.append(type_val)
            
            where_clause = " AND ".join(filters)
            
            query_str = f"""
                SELECT Message, Timestamp FROM rail_data 
                WHERE {where_clause}
                ORDER BY Timestamp 
                LIMIT %s;
            """
            params.append(sample_limit)
            cur.execute(query_str, tuple(params))
            messages_data = cur.fetchall() # List of (Message, Timestamp) tuples
    except psycopg2.Error as e:
        return f"Error fetching messages for VobcId {vobcid}: {e}"
    finally:
        if conn: conn.close()

    if not messages_data:
        return f"No messages found for VobcId {vobcid} with given filters (sample limit {sample_limit})."

    record_nums = []
    for msg_str, ts in messages_data:
        parsed = parse_message_params(msg_str)
        if parsed.get("record_num") is not None:
            record_nums.append({"timestamp": ts, "record_num": parsed["record_num"]})
    
    if not record_nums:
        return f"Could not parse 'record_num' from messages for VobcId {vobcid}."

    df_records = pd.DataFrame(record_nums).sort_values("timestamp")
    df_records["diff"] = df_records["record_num"].diff()
    
    gaps = df_records[(df_records["diff"].notna()) & (df_records["diff"] != 1) & (df_records["diff"] != 0)] # Diff can be 0 if same record num appears
    resets = df_records[df_records["diff"] < 0] # Negative diff suggests a reset

    summary = f"Record Number Analysis for VobcId {vobcid} (sample of {len(messages_data)} messages):\n"
    if not gaps.empty:
        summary += f"Potential Gaps/Jumps (expected diff=1):\n{gaps[['timestamp', 'record_num', 'diff']].head().to_string(index=False)}\n"
    else:
        summary += "No significant gaps found in record_num sequence (based on diff != 1).\n"
    
    if not resets.empty:
        summary += f"Potential Resets (record_num decreased):\n{resets[['timestamp', 'record_num', 'diff']].head().to_string(index=False)}\n"
    else:
        summary += "No obvious resets found in record_num sequence.\n"
        
    return summary

def get_vccnum_healthcode_analysis(limit=10):
    """Analyzes correlation between VccNum and Healthcode/Type patterns."""
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    
    try:
        with conn.cursor() as cur:
            # Fetch data for crosstabulation
            query = """
                SELECT VccNum, Healthcode, COUNT(*) as frequency
                FROM rail_data
                GROUP BY VccNum, Healthcode
                ORDER BY VccNum, frequency DESC;
            """
            # This query can be large. For a chatbot, we need to summarize.
            # Alternative: Focus on VccNum for top Healthcodes or top VccNum by error count.
            # Here's a simplified approach: distribution of top healthcodes per top VccNum.

            # Find top VccNum by event count
            query_top_vcc = """
                SELECT VccNum FROM rail_data GROUP BY VccNum ORDER BY COUNT(*) DESC LIMIT 5;
            """
            cur.execute(query_top_vcc)
            top_vcc_nums = [row[0] for row in cur.fetchall()]

            if not top_vcc_nums:
                return "No VccNum data to analyze."

            results_df = pd.DataFrame()
            for vcc_num in top_vcc_nums:
                query_hc_for_vcc = """
                    SELECT VccNum, Healthcode, COUNT(*) as count_hc
                    FROM rail_data
                    WHERE VccNum = %s
                    GROUP BY VccNum, Healthcode
                    ORDER BY count_hc DESC
                    LIMIT %s; 
                """
                cur.execute(query_hc_for_vcc, (vcc_num, limit)) # Top N healthcodes for this VCC
                rows = cur.fetchall()
                if rows:
                    temp_df = pd.DataFrame(rows, columns=['VccNum', 'Healthcode', 'Frequency'])
                    results_df = pd.concat([results_df, temp_df])
            
            if not results_df.empty:
                return f"Top {limit} Healthcodes for Top VccNums:\n{results_df.to_string(index=False)}"
            return "No significant VccNum to Healthcode patterns found with current query."

    except psycopg2.Error as e:
        return f"Error analyzing VccNum and Healthcodes: {e}"
    finally:
        if conn: conn.close()