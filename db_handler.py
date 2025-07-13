# db_handler.py
import psycopg2
from psycopg2 import sql
import pandas as pd
from config import DB_CONFIG


def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def fetch_healthcode_description(healthcode_str, type_int=None):
    """Fetches description for a given healthcode and optional type."""
    conn = get_db_connection()
    if not conn:
        return "Error: Database connection failed."

    description_parts = []
    try:
        with conn.cursor() as cur:
            query_base = "SELECT Description, TypeDescription FROM HC_Descriptions WHERE Healthcode = %s"
            params = [healthcode_str]
            if type_int is not None:
                query_base += " AND Type = %s"
                params.append(type_int)
            
            cur.execute(query_base, tuple(params))
            results = cur.fetchall()

            if results:
                for row in results:
                    desc = row[0]
                    type_desc = row[1]
                    full_desc = f"Healthcode {healthcode_str}"
                    if type_int is not None:
                        full_desc += f" (Type {type_int})"
                    full_desc += f": {desc}"
                    if type_desc:
                        full_desc += f" (Further details: {type_desc})"
                    description_parts.append(full_desc)
                return "\n".join(description_parts) if description_parts else f"No specific entry for Healthcode {healthcode_str}" + (f" with Type {type_int}." if type_int else ".")
            else:
                return f"No description found for Healthcode {healthcode_str}" + (f" with Type {type_int}." if type_int else ".")
    except psycopg2.Error as e:
        return f"Error fetching healthcode description: {e}"
    finally:
        if conn:
            conn.close()

def fetch_data_for_vobcid(vobcid_int, limit=5):
    """Fetches recent data for a given VobcId."""
    conn = get_db_connection()
    if not conn:
        return "Error: Database connection failed."
    
    try:
        with conn.cursor() as cur:
            # Fetch latest 'Type' and a few recent events for context
            query = sql.SQL("""
                SELECT Timestamp, Healthcode, Type, Message, TrackSection 
                FROM rail_data 
                WHERE VobcId = %s 
                ORDER BY Timestamp DESC 
                LIMIT %s;
            """)
            cur.execute(query, (vobcid_int, limit))
            results = cur.fetchall()
            
            if results:
                # Convert to DataFrame for easy formatting, then to string
                df_results = pd.DataFrame(results, columns=["Timestamp", "Healthcode", "Type", "Message", "TrackSection"])
                latest_type = df_results['Type'].iloc[0] if not df_results.empty else "Unknown"
                context = f"For VobcId {vobcid_int}, the most recent known operational Type is {latest_type}.\n"
                context += "Recent events:\n" + df_results.to_string(index=False)
                return context
            else:
                return f"No data found for VobcId {vobcid_int}."
    except psycopg2.Error as e:
        return f"Error fetching data for VobcId: {e}"
    finally:
        if conn:
            conn.close()

# You can add more specific query functions as needed
# Example: Get the most recent Type for a VobcId
def get_current_type_for_vobcid(vobcid_int):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            query = sql.SQL("""
                SELECT Type 
                FROM rail_data 
                WHERE VobcId = %s 
                ORDER BY Timestamp DESC 
                LIMIT 1;
            """)
            cur.execute(query, (vobcid_int,))
            result = cur.fetchone()
            return f"The current Type for VobcId {vobcid_int} is {result[0]}." if result else f"No Type information found for VobcId {vobcid_int}."
    except psycopg2.Error as e:
        return f"Error fetching type for VobcId: {e}"
    finally:
        if conn: conn.close()
        
        # db_handler.py (Add these functions to your existing file)

def get_frequent_healthcode_type_combinations(limit=10):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            query = """
                SELECT rd.Healthcode, rd.Type, hc.Description, hc.TypeDescription, COUNT(*) as frequency
                FROM rail_data rd
                LEFT JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode AND rd.Type = hc.Type
                GROUP BY rd.Healthcode, rd.Type, hc.Description, hc.TypeDescription
                ORDER BY frequency DESC
                LIMIT %s;
            """
            cur.execute(query, (limit,))
            results = cur.fetchall()
            if results:
                df_results = pd.DataFrame(results, columns=["Healthcode", "Type", "Description", "TypeDescription", "Frequency"])
                return f"Top {limit} most frequent Healthcode/Type combinations:\n{df_results.to_string(index=False)}"
            return "No Healthcode/Type combinations found."
    except psycopg2.Error as e:
        return f"Error fetching frequent healthcodes: {e}"
    finally:
        if conn: conn.close()
        
def get_frequent_healthcode(limit=10):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            query = """
                SELECT rd.Healthcode, rd.Type, hc.Description, hc.TypeDescription, COUNT(*) as frequency
                FROM rail_data rd
                LEFT JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode AND rd.Type = hc.Type
                GROUP BY rd.Healthcode, rd.Type, hc.Description, hc.TypeDescription
                ORDER BY frequency DESC
                LIMIT %s;
            """
            cur.execute(query, (limit,))
            results = cur.fetchall()
            if results:
                df_results = pd.DataFrame(results, columns=["Healthcode", "Type", "Description", "TypeDescription", "Frequency"])
                return f"Top {limit} most frequent Healthcode/Type combinations:\n{df_results.to_string(index=False)}"
            return "No Healthcode/Type combinations found."
    except psycopg2.Error as e:
        return f"Error fetching frequent healthcodes: {e}"
    finally:
        if conn: conn.close()

def get_vobcid_error_counts(limit=5, exclude_hc67=True):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            # Note: 'Error' definition can be refined. Here, not HC '67' or any other list.
            where_clause = "WHERE Healthcode != '67'" if exclude_hc67 else ""
            query_str = f"""
                SELECT VobcId, COUNT(*) as error_count
                FROM rail_data
                {where_clause}
                GROUP BY VobcId
                ORDER BY error_count DESC
                LIMIT %s;
            """
            cur.execute(query_str, (limit,))
            results = cur.fetchall()
            if results:
                df_results = pd.DataFrame(results, columns=["VobcId", "ErrorCount"])
                return f"Top {limit} VobcIds reporting most errors (excluding HC67: {exclude_hc67}):\n{df_results.to_string(index=False)}"
            return "No VobcId error counts found."
    except psycopg2.Error as e:
        return f"Error fetching VobcId error counts: {e}"
    finally:
        if conn: conn.close()

def get_temporal_error_patterns(healthcode_filter=None, type_filter=None, time_granularity='hour', limit=10):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            time_extract_part = ""
            if time_granularity == 'hour':
                time_extract_part = "EXTRACT(HOUR FROM Timestamp)"
            elif time_granularity == 'day_of_week': # 0 = Sunday, 6 = Saturday for PostgreSQL EXTRACT(DOW ...)
                time_extract_part = "EXTRACT(DOW FROM Timestamp)"
            elif time_granularity == 'date':
                time_extract_part = "DATE(Timestamp)"
            else:
                return "Error: Invalid time_granularity. Choose 'hour', 'day_of_week', or 'date'."

            filters = []
            params = []
            if healthcode_filter:
                filters.append("rd.Healthcode = %s")
                params.append(healthcode_filter)
            if type_filter is not None: # Type can be 0
                filters.append("rd.Type = %s")
                params.append(type_filter)
            
            where_clause = ""
            if filters:
                where_clause = "WHERE " + " AND ".join(filters)

            query_str = f"""
                SELECT {time_extract_part} as time_period, rd.Healthcode, rd.Type, COUNT(*) as frequency
                FROM rail_data rd
                {where_clause}
                GROUP BY time_period, rd.Healthcode, rd.Type
                ORDER BY frequency DESC
                LIMIT %s;
            """
            params.append(limit)
            cur.execute(query_str, tuple(params))
            results = cur.fetchall()
            if results:
                df_results = pd.DataFrame(results, columns=[f"TimePeriod ({time_granularity})", "Healthcode", "Type", "Frequency"])
                return f"Top error patterns by {time_granularity}:\n{df_results.to_string(index=False)}"
            return f"No specific temporal patterns found for the given filters."
    except psycopg2.Error as e:
        return f"Error fetching temporal patterns: {e}"
    finally:
        if conn: conn.close()

def get_engineering_train_events_summary():
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            query = """
                SELECT rd.Timestamp, rd.VobcId, rd.TrackSection, rd.Healthcode, rd.Type, hc.Description
                FROM rail_data rd
                JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode AND rd.Type = hc.Type
                WHERE hc.EngineeringTrainOnly = 1
                ORDER BY rd.Timestamp DESC
                LIMIT 20; -- Limit for brevity in chatbot
            """
            cur.execute(query)
            results = cur.fetchall()
            if results:
                df_results = pd.DataFrame(results, columns=["Timestamp", "VobcId", "TrackSection", "Healthcode", "Type", "Description"])
                count = len(df_results) # This is only for the limited sample, full count would need another query
                return f"Found events related to Healthcodes flagged 'EngineeringTrainOnly'. Recent examples (up to 20):\n{df_results.to_string(index=False)}\nNote: Correlation with actual engineering schedules requires external data not available to me."
            return "No events found for Healthcodes flagged 'EngineeringTrainOnly'."
    except psycopg2.Error as e:
        return f"Error fetching engineering train events: {e}"
    finally:
        if conn: conn.close()

def get_excludable_errors_summary():
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            # Get total count of excludable errors
            query_count = """
                SELECT COUNT(rd.*)
                FROM rail_data rd
                JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode AND rd.Type = hc.Type
                WHERE hc.Exclude = 1;
            """
            cur.execute(query_count)
            total_excludable_count = cur.fetchone()[0]

            # Get summary of types of excludable errors
            query_summary = """
                SELECT rd.Healthcode, rd.Type, hc.Description, COUNT(*) as frequency
                FROM rail_data rd
                JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode AND rd.Type = hc.Type
                WHERE hc.Exclude = 1
                GROUP BY rd.Healthcode, rd.Type, hc.Description
                ORDER BY frequency DESC
                LIMIT 10;
            """
            cur.execute(query_summary)
            results = cur.fetchall()
            if total_excludable_count > 0 and results:
                df_results = pd.DataFrame(results, columns=["Healthcode", "Type", "Description", "Frequency"])
                return f"Total excludable errors found: {total_excludable_count}.\nTop {len(df_results)} types of excludable errors:\n{df_results.to_string(index=False)}"
            elif total_excludable_count > 0:
                return f"Total excludable errors found: {total_excludable_count}. No further breakdown available or summary limit reached."
            return "No events found for Healthcodes flagged 'Exclude=1'."
    except psycopg2.Error as e:
        return f"Error fetching excludable errors: {e}"
    finally:
        if conn: conn.close()

def get_location_specific_error_counts(location_field='TrackSection', limit_per_location=5, top_locations=5):
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    if location_field not in ['TrackSection', 'SegmentId', 'LoopNum', 'XOverNum']:
        return "Error: Invalid location_field specified."
    try:
        with conn.cursor() as cur:
            # Using sql.Identifier for safe dynamic column names
            location_col_sql = sql.Identifier(location_field)
            
            # Find top locations by total error count first
            top_locations_query_str = sql.SQL("""
                SELECT {loc_field}
                FROM rail_data
                WHERE Healthcode != '67' -- Example: exclude common 'normal' code
                GROUP BY {loc_field}
                ORDER BY COUNT(*) DESC
                LIMIT %s;
            """).format(loc_field=location_col_sql)
            cur.execute(top_locations_query_str, (top_locations,))
            top_locs = [row[0] for row in cur.fetchall()]

            if not top_locs:
                return f"No significant activity found to analyze for {location_field}."

            all_results_df = pd.DataFrame()
            for loc_val in top_locs:
                query_str = sql.SQL("""
                    SELECT rd.{loc_field}, rd.Healthcode, hc.Description, COUNT(*) as frequency
                    FROM rail_data rd
                    LEFT JOIN HC_Descriptions hc ON rd.Healthcode = hc.Healthcode -- Simplified join for general description
                    WHERE rd.{loc_field} = %s AND rd.Healthcode != '67'
                    GROUP BY rd.{loc_field}, rd.Healthcode, hc.Description
                    ORDER BY frequency DESC
                    LIMIT %s;
                """).format(loc_field=location_col_sql)
                cur.execute(query_str, (loc_val, limit_per_location))
                results = cur.fetchall()
                if results:
                    df_temp = pd.DataFrame(results, columns=[location_field, "Healthcode", "Description", "Frequency"])
                    all_results_df = pd.concat([all_results_df, df_temp])
            
            if not all_results_df.empty:
                return f"Top error counts for top {top_locations} {location_field}s (HC67 excluded, up to {limit_per_location} HCs per location):\n{all_results_df.to_string(index=False)}"
            return f"No specific error patterns found for top {location_field}s."
    except psycopg2.Error as e:
        return f"Error fetching location-specific errors for {location_field}: {e}"
    finally:
        if conn: conn.close()

def get_ingestion_lag_stats():
    conn = get_db_connection()
    if not conn: return "Error: Database connection failed."
    try:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    MIN(IngestionDate - Timestamp) as min_lag,
                    MAX(IngestionDate - Timestamp) as max_lag,
                    AVG(IngestionDate - Timestamp) as avg_lag,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY IngestionDate - Timestamp) as median_lag
                FROM rail_data
                WHERE IngestionDate IS NOT NULL AND Timestamp IS NOT NULL; 
            """
            # Note: AVG and PERCENTILE_CONT on intervals can be complex in SQL.
            # It might be better to fetch raw differences and calculate in Pandas for more complex stats.
            # This query provides basic interval stats.
            cur.execute(query)
            results = cur.fetchone()
            if results:
                return f"Data Ingestion Lag Statistics:\nMin Lag: {results[0]}\nMax Lag: {results[1]}\nAvg Lag: {results[2]}\nMedian Lag: {results[3]}"
            return "Could not calculate ingestion lag statistics."
    except psycopg2.Error as e:
        return f"Error fetching ingestion lag: {e}"
    finally:
        if conn: conn.close()