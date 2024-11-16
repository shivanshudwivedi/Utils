from flask import Flask, request, jsonify
from vision_processor import VisionProcessor
import json
import os
from datetime import datetime
from threading import Thread
from flask_cors import CORS

app = Flask(__name__)
vision_processor = None
processing_results = None

CORS(app)

def initialize_vision():
    global vision_processor
    if vision_processor is None:
        vision_processor = VisionProcessor()

initialize_vision()

@app.route('/start_vision_processing', methods=['POST'])
def start_vision_processing():
    """Start the vision processing in a separate thread"""
    global processing_results
    try:
        def process_vision():
            global processing_results
            try:
                objects_data = vision_processor.process_objects()
                vision_processor.save_results(objects_data)
                processing_results = objects_data
            except Exception as e:
                print(f"Error in vision processing thread: {str(e)}")
                processing_results = None

        processing_results = None  # Reset results
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
        if processing_results is None:
            return jsonify({
                "status": "processing",
                "message": "Results not yet available",
                "processed_data": []
            })

        return jsonify({
            "status": "success",
            "message": "Results retrieved successfully",
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