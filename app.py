import os
import torch
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import easyocr

# Import your refactored inference code
from inference_module import SimpleYOLO, YOLOConfig, process_single_image

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'yolo_output_pytorch/best_model.pth'
PARAMS_PATH = 'yolo_output_pytorch/best_params.json'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.secret_key = 'a_different_super_secret_key' # Required for flash messages

# --- Pre-load Models (for performance) ---
print("Loading models and setting up device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
ocr_reader = None

try:
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    config = YOLOConfig(
        image_size=best_params['image_size'],
        grid_size=best_params['grid_size']
    )
    model = SimpleYOLO(grid_size=config.grid_size, image_size=config.image_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("Models loaded successfully!")
except Exception as e:
    print(f"FATAL ERROR loading models: {e}")
    # The app will still run but show an error on the page.

# --- Helper Function ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    # Check if models failed to load during startup
    if model is None or ocr_reader is None:
        flash("Application Error: Models are not loaded. Please check server logs.", "error")
        return render_template('index.html')

    if request.method == 'POST':
        # --- This section handles the file upload and is crucial for fixing the bug ---
        
        # 1. Check if the request has a file part
        if 'file' not in request.files:
            flash('No file part in the request. This is unusual.', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 2. Check if the user submitted an empty file input
        if file.filename == '':
            flash('No file selected. Please choose an image.', 'error')
            return redirect(request.url)
        
        # 3. Check if the file is valid and has an allowed extension
        if file and allowed_file(file.filename):
            # Ensure the upload and results directories exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            print(f"Image saved to: {upload_path}")

            # --- Process the image to get the results ---
            analysis_result = process_single_image(
                image_path=upload_path,
                model=model,
                config=config,
                ocr_reader=ocr_reader,
                device=device,
                output_dir_results=app.config['RESULTS_FOLDER']
            )
            
            # --- Render the result page with the extracted data ---
            return render_template('result.html', result=analysis_result)
        else:
            # 4. If the file extension is not allowed
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(request.url)

    # If it's a GET request, just show the upload page
    return render_template('index.html')

if __name__ == '__main__':
    # This block will only run when you execute "python app.py"
    # It will not run when Gunicorn or Waitress imports the app
    app.run(debug=True)