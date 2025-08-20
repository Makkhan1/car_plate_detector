import numpy as np
import cv2
import os
import json
import torch
import torch.nn as nn
from torchvision.transforms import transforms
import easyocr
from dataclasses import dataclass
import exifread
import traceback
from PIL import Image # Use PIL for saving images to avoid CV2 color issues

# --- 1. Define Model Architecture (Same as your original script) ---
class SimpleYOLO(nn.Module):
    def __init__(self, input_channels=3, grid_size=7, num_outputs=5, dropout_rate=0.5, image_size=416):
        super(SimpleYOLO, self).__init__()
        self.grid_size = grid_size
        self.num_outputs = num_outputs
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self._get_conv_output_size(), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.grid_size * self.grid_size * self.num_outputs),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.image_size, self.image_size)
            output = self.features(dummy_input)
            return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_outputs)
        return x

# --- 2. Define YOLOConfig (Same as your original script) ---
@dataclass
class YOLOConfig:
    image_size: int
    grid_size: int
    output_size: int = 5

# --- 3. Helper Functions (Slightly modified for web app use) ---

def gps_from_exifread(img_path):
    lat, lon = None, None
    try:
        with open(img_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        
        def _coord(tag_name):
            if tag_name not in tags: return None
            ref_tag_name = tag_name + 'Ref'
            if ref_tag_name not in tags: return None
            ref = str(tags[ref_tag_name])
            deg_val, min_val, sec_val = tags[tag_name].values
            deg = float(deg_val.num) / float(deg_val.den)
            min_ = float(min_val.num) / float(min_val.den)
            sec = float(sec_val.num) / float(sec_val.den)
            decimal_degrees = deg + min_ / 60 + sec / 3600
            sign = -1 if ref in ['S','W'] else 1
            return sign * decimal_degrees

        lat = _coord('GPS GPSLatitude')
        lon = _coord('GPS GPSLongitude')
    except Exception:
        pass # Fail silently for web app
    return lat, lon

def process_single_image(image_path, model, config, ocr_reader, device, output_dir_results):
    """
    Processes a single image, saves annotated and blurred versions, and returns results.
    This is the core function the web app will call.
    """
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"error": f"Could not load image: {image_path}"}
            
        original_height, original_width = original_image.shape[:2]
        
        # --- Geotagging ---
        latitude, longitude = gps_from_exifread(image_path)
        geotag_str = f"Lat: {latitude:.4f}, Lon: {longitude:.4f}" if latitude is not None else "No Geotag Found"

        # --- YOLO Inference ---
        transform_for_inference = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
        yolo_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        yolo_image = transform_for_inference(yolo_image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            yolo_predictions = model(yolo_image).squeeze(0).cpu().numpy()

        best_confidence = 0
        best_bbox_norm = None
        for i in range(config.grid_size):
            for j in range(config.grid_size):
                confidence = yolo_predictions[i, j, 0]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_bbox_norm = yolo_predictions[i, j, 1:]
        
        detection_threshold = 0.5
        license_text = "N/A"
        bbox_coords = None
        
        # --- BBox and OCR Processing ---
        if best_bbox_norm is not None and best_confidence > detection_threshold:
            x_center, y_center, width, height = np.clip(best_bbox_norm, 0, 1)
            abs_x_center, abs_y_center = int(x_center * original_width), int(y_center * original_height)
            abs_w, abs_h = int(width * original_width), int(height * original_height)
            
            padding_x, padding_y = int(abs_w * 0.15), int(abs_h * 0.25)
            x_min = max(0, abs_x_center - abs_w // 2 - padding_x)
            y_min = max(0, abs_y_center - abs_h // 2 - padding_y)
            x_max = min(original_width, abs_x_center + abs_w // 2 + padding_x)
            y_max = min(original_height, abs_y_center + abs_h // 2 + padding_y)
            
            if x_min < x_max and y_min < y_max:
                bbox_coords = (x_min, y_min, x_max, y_max)
                license_plate = original_image[y_min:y_max, x_min:x_max]
                
                if license_plate.size > 0:
                    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    result = ocr_reader.readtext(gray, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                    if result:
                        license_text = ''.join([box[1] for box in result]).replace(" ", "").upper()

        # --- Create and Save Result Images ---
        filename = os.path.basename(image_path)
        annotated_image = original_image.copy()
        blurred_image = original_image.copy()

        if bbox_coords:
            x1, y1, x2, y2 = bbox_coords
            # Draw bounding box on annotated image
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_image, f"LP: {license_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Create blurred version
            roi = blurred_image[y1:y2, x1:x2]
            if roi.size > 0:
                kernel_size = max(1, int(roi.shape[1] / 4)) | 1 # Ensure odd kernel
                blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                blurred_image[y1:y2, x1:x2] = blurred_roi

        # Save images
        annotated_filename = f"annotated_{filename}"
        blurred_filename = f"blurred_{filename}"
        annotated_save_path = os.path.join(output_dir_results, annotated_filename)
        blurred_save_path = os.path.join(output_dir_results, blurred_filename)

        # Use PIL to save to avoid BGR/RGB issues when displaying in HTML
        Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(annotated_save_path)
        Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)).save(blurred_save_path)

        # Return a dictionary with all the results
        return {
            "license_text": license_text,
            "geotag_str": geotag_str,
            "confidence": f"{best_confidence:.2f}",
            "annotated_image_url": os.path.join('results', annotated_filename).replace("\\", "/"),
            "blurred_image_url": os.path.join('results', blurred_filename).replace("\\", "/"),
            "original_image_url": os.path.join('uploads', filename).replace("\\", "/"),
            "error": None
        }
            
    except Exception as e:
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {str(e)}"}