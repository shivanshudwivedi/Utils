from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO
import cv2
import pytesseract
import os

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = YOLO("yolov5n.pt")  # Use "yolov5s" for a small, fast model

# Path to Tesseract executable (required for Windows)
# Uncomment and specify if using Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Function to perform OCR on an image
def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Generator function to process frames from the camera
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Object Detection with YOLOv5
        results = model.predict(source=frame, verbose=False)
        for box, class_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(class_id)]
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the format required for Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to capture a picture and perform OCR
@app.route('/capture', methods=['POST'])
def capture_image():
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500

    # Object Detection with YOLOv5
    results = model.predict(source=frame, verbose=False)
    detected_objects = []
    for box, class_id, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(class_id)]

        # Crop the detected object for OCR
        cropped_object = frame[y1:y2, x1:x2]
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
        <title>YOLOv5 + OCR Real-Time Object Detection</title>
    </head>
    <body>
        <h1>YOLOv5 + OCR Real-Time Object Detection</h1>
        <p>Live Video Feed:</p>
        <img src="/video_feed" style="width: 600px; height: auto;"/>
        <br><br>
        <button onclick="captureImage()">Take Picture</button>
        <p id="result"></p>
        <script>
            async function captureImage() {
                const response = await fetch('/capture', { method: 'POST' });
                const result = await response.json();
                document.getElementById('result').textContent = JSON.stringify(result, null, 2);
            }
        </script>
    </body>
    </html>
    '''

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
