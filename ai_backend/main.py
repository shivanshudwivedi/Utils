from flask import Flask, request, jsonify
from vision_processor import VisionProcessor
import json
import os
from datetime import datetime
from threading import Thread, Event
from flask_cors import CORS

app = Flask(__name__)
vision_processor = None
processing_results = None
processing_event = Event()

CORS(app)

def initialize_vision():
    global vision_processor
    if vision_processor is None:
        vision_processor = VisionProcessor()

initialize_vision()

@app.route('/start_vision_processing', methods=['POST'])
def start_vision_processing():
    """Start the vision processing in a separate thread"""
    global processing_results, processing_event
    try:
        def process_vision():
            global processing_results
            try:
                # Process until we find a high-confidence detection
                objects_data = vision_processor.process_objects(confidence_threshold=0.7)
                if objects_data:
                    vision_processor.save_results(objects_data)
                processing_results = objects_data
                processing_event.set()  # Signal that processing is complete
            except Exception as e:
                print(f"Error in vision processing thread: {str(e)}")
                processing_results = []
                processing_event.set()

        processing_results = None  # Reset results
        processing_event.clear()  # Reset event
        thread = Thread(target=process_vision)
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Vision processing started"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/process_vision_results', methods=['GET'])
def process_vision_results():
    """Retrieve the processed vision results"""
    try:
        # Check if processing is complete
        if not processing_event.is_set():
            return jsonify({
                "status": "processing",
                "message": "Still searching for high-confidence detection",
                "processed_data": []
            })

        if not processing_results:
            return jsonify({
                "status": "complete",
                "message": "No high-confidence detections found",
                "processed_data": []
            })

        return jsonify({
            "status": "success",
            "message": "High-confidence detection found",
            "processed_data": processing_results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "processed_data": []
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "vision_processor": "initialized" if vision_processor else "not initialized",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)