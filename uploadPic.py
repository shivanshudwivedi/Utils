from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = YOLO("yolov5n.pt")  # Use "yolov5s.pt" for better accuracy if available

# Path to Tesseract executable (required for Windows)
# Uncomment and specify if using Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Function to perform OCR on a detected object
def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Route for uploading an image and processing it
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the uploaded image
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    image_np = np.array(image)

    # Object Detection with YOLOv5
    results = model.predict(source=image_np, verbose=False)
    detected_objects = []
    for box, class_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(class_id)]

        # Crop the detected object for OCR
        cropped_object = image_np[y1:y2, x1:x2]
        text = perform_ocr(cropped_object)

        detected_objects.append({
            "label": label,
            "confidence": float(conf),
            "bounding_box": [x1, y1, x2, y2],
            "text": text if text else "No text detected"
        })

    return jsonify({"detected_objects": detected_objects})

# Home route for rendering the HTML interface
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>YOLOv5 + OCR Object Detection</title>
    </head>
    <body>
        <h1>Upload an Image for Object Detection and OCR</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="image">Select an image:</label>
            <input type="file" id="image" name="image" accept="image/*" required>
            <button type="submit">Upload and Process</button>
        </form>
        <p id="result"></p>
    </body>
    </html>
    '''

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
