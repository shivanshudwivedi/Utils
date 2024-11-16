import torch
import torchvision
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2

class ObjectDetectionModel:
    def __init__(self):
        # Load pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # COCO dataset class names
        self.CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def process_image(self, image_bytes):
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Transform image
        image_tensor = self.transform(image)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model([image_tensor])
            
        # Process results
        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        
        # Filter predictions with confidence > 0.5
        mask = scores > 0.5
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]
        
        results = []
        for box, label, score in zip(boxes, labels, scores):
            results.append({
                'label': self.CLASSES[label],
                'confidence': float(score),
                'bbox': box.tolist()
            })
            
        return results
