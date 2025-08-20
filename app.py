import os
import torch
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import easyocr

# Import your refactored inference code from the other file
from inference_module import SimpleYOLO, YOLOConfig, process_single_image

# =====================================================================================
# --- Configuration: All settings are here for easy access ---
# =====================================================================================

# Flask settings for file uploads
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# This is the local directory on the Render server where files will be downloaded
ASSETS_DIR = 'models' 

# --- IMPORTANT: PASTE YOUR HUGGING FACE URLS HERE ---
# Get these by going to your file on Hugging Face, right-clicking the "download" button,
# and selecting "Copy Link Address".

# 1. Model file configuration
MODEL_FILENAME = 'best_model.pth'
MODEL_PATH = os.path.join(ASSETS_DIR, MODEL_FILENAME)
MODEL_URL = 'https://huggingface.co/YourUsername/your-repo-name/resolve/main/best_model.pth'

# 2. Parameters file configuration
PARAMS_FILENAME = 'best_params.json'
PARAMS_PATH = os.path.join(ASSETS_DIR, PARAMS_FILENAME)
PARAMS_URL = 'https://huggingface.co/YourUsername/your-repo-name/resolve/main/best_params.json'
# ---------------------------------------------------------

# Initialize the Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'a_super_secret_key_for_flash_messages' # Required for flash messages

# =====================================================================================
# --- Helper Function for Downloading Files ---
# =====================================================================================

def download_file_if_needed(url, path, file_description):
    """
    Downloads a file from a URL to a local path if it doesn't already exist.
    This is crucial for fetching models from the cloud on server startup.
    """
    if not os.path.exists(path):
        print(f"{file_description} not found at '{path}'. Downloading from URL...")
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes (like 404)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{file_description} downloaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"FATAL: Error downloading {file_description}: {e}")
            # Stop the application if a critical file can't be downloaded
            raise
    else:
        print(f"{file_description} already exists at '{path}'. Skipping download.")

# =====================================================================================
# --- Model Loading: This runs ONCE when the server starts ---
# =====================================================================================

print("--- Initializing Application ---")
# Force CPU device for compatibility with Render's servers
device = torch.device("cpu")
model = None
ocr_reader = None
config = None

try:
    # Step 1: Download the necessary files from the cloud
    print("Step 1: Checking for model and parameter files...")
    download_file_if_needed(PARAMS_URL, PARAMS_PATH, "Parameters file")
    download_file_if_needed(MODEL_URL, MODEL_PATH, "PyTorch model file")

    # Step 2: Load the downloaded files into the application
    print("\nStep 2: Loading models into memory...")
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    
    config = YOLOConfig(
        image_size=best_params['image_size'],
        grid_size=best_params['grid_size']
    )
    
    # Load the PyTorch model
    model = SimpleYOLO(grid_size=config.grid_size, image_size=config.image_size).to(device)
    # map_location=device is CRITICAL for loading a GPU-trained model onto a CPU server
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Load the EasyOCR reader
    ocr_reader = easyocr.Reader(['en'], gpu=False) # gpu=False is essential for CPU-only servers
    
    print("\n--- Models loaded successfully! Application is ready. ---")

except Exception as e:
    # If anything goes wrong during startup, log the error.
    # The app will still run, but the routes will show an error message.
    print(f"FATAL ERROR during model initialization: {e}")


# =====================================================================================
# --- Flask Routes ---
# =====================================================================================

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    # Check if models failed to load during startup
    if model is None or ocr_reader is None:
        flash("CRITICAL ERROR: The machine learning models could not be loaded. Please check the server logs for more details.", "error")
        return render_template('index.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image to upload.', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the upload and results directories exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
            
            file.save(upload_path)
            print(f"Image saved to: {upload_path}")

            # Process the image using the function from inference_module.py
            analysis_result = process_single_image(
                image_path=upload_path,
                model=model,
                config=config,
                ocr_reader=ocr_reader,
                device=device,
                output_dir_results=app.config['RESULTS_FOLDER']
            )
            
            # Render the result page with the extracted data
            return render_template('result.html', result=analysis_result)
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(request.url)

    # For a GET request, just show the main upload page
    return render_template('index.html')

# This block is for local testing only. Gunicorn will run the app on Render.
if __name__ == '__main__':
    app.run(debug=True)
