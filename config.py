# PostgreSQL Database Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "pdudas",
    "password": "password",
    "host": "localhost",  # Or your DB host
    "port": "5432"       # Or your DB port
}

# Ollama API Configuration
OLLAMA_API_ENDPOINT = "http://localhost:11434/api/generate" # Default Ollama endpoint
MISTRAL_MODEL_NAME = "mistral:instruct" # Or the specific model tag you use

# File Paths
CAUSAL_SUMMARY_FILE = "causal_analysis_summary.txt" # Path to your summary file
PRED_MAINT_SCRIPT_PATH = "predictive_maintenance.py" # Full path to the script
PRED_MAINT_OUTPUT_TXT = "pred_maint_output.txt" # Where the script saves its text output
PRED_MAINT_IMAGE_DIR = "pred_maint_images/" # Directory where images are saved