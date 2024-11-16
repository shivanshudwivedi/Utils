"use client"
import React, { useRef, useEffect, useState } from 'react';


const ObjectDetectionScanner = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);

  // Start the camera stream
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setError(null);
      }
    } catch (err) {
      setError('Unable to access camera. Please ensure you have granted camera permissions.');
      console.error('Error accessing camera:', err);
    }
  };

  // Capture and process frame
  const processFrame = async () => {
    if (videoRef.current && canvasRef.current && !isProcessing) {
      setIsProcessing(true);
      
      try {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        // Set canvas size to match video
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        
        // Draw current frame to canvas
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
        
        // Create form data
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        // Send to backend
        const response = await fetch('http://localhost:8000/detect', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) throw new Error('Failed to process image');
        
        const data = await response.json();
        setDetections(data.detections);
        
        // Draw bounding boxes
        data.detections.forEach(detection => {
          const [x1, y1, x2, y2] = detection.bbox;
          context.strokeStyle = '#22c55e';
          context.lineWidth = 2;
          context.strokeRect(x1, y1, x2 - x1, y2 - y1);
          
          // Draw label background
          context.fillStyle = 'rgba(34, 197, 94, 0.9)';
          const text = `${detection.label} ${(detection.confidence * 100).toFixed(1)}%`;
          const textWidth = context.measureText(text).width;
          context.fillRect(x1, y1 - 25, textWidth + 10, 20);
          
          // Draw label text
          context.fillStyle = '#ffffff';
          context.font = '14px sans-serif';
          context.fillText(text, x1 + 5, y1 - 10);
        });
        
      } catch (err) {
        console.error('Error processing frame:', err);
        setError('Failed to process image. Please try again.');
      }
      
      setIsProcessing(false);
    }
  };

  // Initialize camera on component mount
  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="max-w-2xl mx-auto p-4">
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        {/* Header */}
        <div className="p-4 bg-gray-50 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg 
              className="w-6 h-6 text-gray-700" 
              fill="none" 
              strokeWidth={1.5} 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0zM18.75 10.5h.008v.008h-.008V10.5z" />
            </svg>
            <h2 className="text-lg font-semibold text-gray-900">Object Scanner</h2>
          </div>
          <button
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isProcessing || !isStreaming
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-green-500 hover:bg-green-600 text-white'
            }`}
            onClick={processFrame}
            disabled={isProcessing || !isStreaming}
          >
            {isProcessing ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Processing...</span>
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" strokeWidth={1.5} stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span>Scan</span>
              </>
            )}
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {error ? (
            <div className="bg-red-50 text-red-700 p-4 rounded-lg mb-4">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5" fill="none" strokeWidth={1.5} stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                <p className="font-medium">{error}</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Video feed and canvas container */}
              <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  className="w-full h-full object-cover"
                  autoPlay
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 w-full h-full"
                />
              </div>
              
              {/* Detection results */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-lg font-semibold mb-3">Detection Results</h3>
                {detections.length > 0 ? (
                  <div className="space-y-2">
                    {detections.map((detection, index) => (
                      <div 
                        key={index} 
                        className="flex justify-between items-center py-2 px-3 bg-white rounded-lg shadow-sm"
                      >
                        <span className="font-medium text-gray-900">
                          {detection.label}
                        </span>
                        <span className="text-sm text-gray-500">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-4">
                    Click the Scan button to detect objects...
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionScanner;