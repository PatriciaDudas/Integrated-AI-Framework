# pred_maint_interfacer.py
import os
from config import PRED_MAINT_OUTPUT_TXT, PRED_MAINT_IMAGE_DIR # Ensure these are correctly set in config.py

def get_predictive_maintenance_results():
    """
    Reads the pre-existing text output from the predictive maintenance analysis
    and lists any associated images.
    It does NOT run any external script.
    """
    text_output = f"Predictive maintenance results from {PRED_MAINT_OUTPUT_TXT}:\n"
    images_info = ""
    
    # Read the text output from the file generated by the predictive maintenance script
    if os.path.exists(PRED_MAINT_OUTPUT_TXT):
        try:
            with open(PRED_MAINT_OUTPUT_TXT, 'r', encoding='utf-8') as f:
                text_output += f.read()
            if not text_output.strip(): # Check if file is empty
                 text_output = f"The predictive maintenance output file ({PRED_MAINT_OUTPUT_TXT}) was found but is empty."
        except Exception as e:
            error_message = f"Error reading predictive maintenance output file ({PRED_MAINT_OUTPUT_TXT}): {e}"
            print(error_message)
            text_output = error_message
    else:
        text_output = f"Predictive maintenance output file ({PRED_MAINT_OUTPUT_TXT}) not found. The results may not be up-to-date or available."
        print(f"Warning: {PRED_MAINT_OUTPUT_TXT} not found.")
    
    # List images in the output directory (assuming these are static artifacts from a previous run)
    generated_images_paths = []
    if PRED_MAINT_IMAGE_DIR and os.path.isdir(PRED_MAINT_IMAGE_DIR):
        try:
            generated_images_paths = [
                os.path.join(PRED_MAINT_IMAGE_DIR, f) 
                for f in os.listdir(PRED_MAINT_IMAGE_DIR) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
            ]
        except Exception as e:
            print(f"Error listing images in {PRED_MAINT_IMAGE_DIR}: {e}")
            
    if generated_images_paths:
        images_info = f"\nAssociated predictive maintenance images found in '{PRED_MAINT_IMAGE_DIR}': {len(generated_images_paths)} image(s)."
        # You could list the image names if desired:
        # images_info += "\nImage files: " + ", ".join([os.path.basename(p) for p in generated_images_paths])
        print(f"Found images: {generated_images_paths}")
    else:
        images_info = f"\nNo associated predictive maintenance images found in '{PRED_MAINT_IMAGE_DIR}'."

    full_report = text_output + images_info
    return full_report


# # pred_maint_interfacer.py
# import subprocess
# import os
# from config import PRED_MAINT_SCRIPT_PATH, PRED_MAINT_OUTPUT_TXT, PRED_MAINT_IMAGE_DIR

# def run_predictive_maintenance(vobcid=None):
#     """
#     Runs the external predictive maintenance script and returns its text output
#     and information about generated images.
#     """
#     if not os.path.exists(PRED_MAINT_SCRIPT_PATH):
#         return "Error: Predictive maintenance script not found.", []

#     command = ["python", PRED_MAINT_SCRIPT_PATH]
#     if vobcid:
#         command.extend(["--vobcid", str(vobcid)]) # Example: pass VobcId as an argument

#     try:
#         print(f"Running predictive maintenance script: {' '.join(command)}")
#         # Ensure the script writes its output to PRED_MAINT_OUTPUT_TXT
#         # and images to PRED_MAINT_IMAGE_DIR
        
#         # Create output dir if it doesn't exist
#         if PRED_MAINT_IMAGE_DIR and not os.path.exists(PRED_MAINT_IMAGE_DIR):
#             os.makedirs(PRED_MAINT_IMAGE_DIR)
            
#         # Delete old output text file to ensure we read the new one
#         if os.path.exists(PRED_MAINT_OUTPUT_TXT):
#             os.remove(PRED_MAINT_OUTPUT_TXT)

#         process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300) # 5 min timeout

#         if process.returncode != 0:
#             error_message = f"Predictive maintenance script failed with error:\n{process.stderr}"
#             print(error_message)
#             return error_message, []

#         # Read the text output from the file generated by the script
#         if os.path.exists(PRED_MAINT_OUTPUT_TXT):
#             with open(PRED_MAINT_OUTPUT_TXT, 'r', encoding='utf-8') as f:
#                 text_output = f.read()
#         else:
#             text_output = "Predictive maintenance script ran, but no text output file was found."
#             print(f"Warning: {PRED_MAINT_OUTPUT_TXT} not found after script execution.")
        
#         # List images in the output directory (assuming the script places them there)
#         generated_images = []
#         if PRED_MAINT_IMAGE_DIR and os.path.isdir(PRED_MAINT_IMAGE_DIR):
#             generated_images = [os.path.join(PRED_MAINT_IMAGE_DIR, f) for f in os.listdir(PRED_MAINT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        
#         image_message = ""
#         if generated_images:
#             image_message = f"\nThe script also generated {len(generated_images)} image(s) which can be found in '{PRED_MAINT_IMAGE_DIR}'."
#             # For a CLI, you can't display images directly. For a web app, you could return paths.
#             print(f"Generated images: {generated_images}")

#         return text_output + image_message, generated_images

#     except subprocess.TimeoutExpired:
#         error_message = "Predictive maintenance script timed out."
#         print(error_message)
#         return error_message, []
#     except Exception as e:
#         error_message = f"Error running predictive maintenance script: {e}"
#         print(error_message)
#         return error_message, []