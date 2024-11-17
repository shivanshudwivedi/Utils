import cv2
from ultralytics import YOLO
import easyocr
import os
import numpy as np
import json
from collections import Counter
import re
import platform
import time

class VisionProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.model = YOLO("yolov8n-seg.pt")
        self.model.overrides["conf"] = 0.5  # Base confidence threshold
        self.model.overrides["iou"] = 0.5
        self.output_dir = "detected_objects"
        self.HUMAN_COLOR = (0, 255, 0)
        self.NON_HUMAN_COLOR = (255, 0, 0)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def process_objects(self, confidence_threshold=0.60):
        """Main processing loop to detect and analyze objects until finding one with sufficient confidence"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Cannot access webcam")

        objects_data = []
        max_attempts = 100  # Maximum number of frames to process
        attempt = 0

        try:
            while attempt < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.model(frame)
                non_human_objects = self._get_non_human_objects(results[0])

                # Process non-human objects
                for mask, bbox, confidence, label in non_human_objects:
                    # Check if confidence meets our threshold
                    if confidence > confidence_threshold:
                        # Process this high-confidence object
                        object_data = self._process_single_object(
                            frame, mask, bbox, confidence, label, 0
                        )
                        if object_data:
                            objects_data = [object_data]  # Keep only this result
                            # Save frame with detection
                            self._save_detection_frame(frame, results[0])
                            return objects_data  # Exit immediately after finding a good match

                attempt += 1
                # Save frame for debugging
                self._display_frame(frame, results[0])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return objects_data  # Will be empty if no high-confidence detection was found

    def _get_non_human_objects(self, result):
        """Extract non-human objects from YOLO results"""
        non_human_objects = []
        for i, box_data in enumerate(result.boxes.data.tolist()):
            x1, y1, x2, y2, confidence, class_id = box_data
            label = self.model.names[int(class_id)]
            
            if label != "person":
                mask = result.masks.data[i]
                bbox = (x1, y1, x2, y2)
                non_human_objects.append((mask, bbox, confidence, label))
        
        return non_human_objects

    def _process_single_object(self, frame, mask, bbox, confidence, label, index):
        """Process a single detected object"""
        try:
            # Save object image
            object_path = self._save_object_image(frame, mask, index)
            
            # Extract text
            extracted_text = self._extract_text(object_path)

            # Create object data
            x1, y1, x2, y2 = bbox
            return {
                "index": index,
                "bounding_box": {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2)
                },
                "confidence": round(float(confidence), 2),
                "label": label,
                "text": extracted_text,
                "image_path": object_path,
                "timestamp": int(time.time())
            }
        except Exception as e:
            print(f"Error processing object {index}: {str(e)}")
            return None

    def _save_object_image(self, frame, mask, index):
        """Save detected object image with background removed"""
        mask = mask.cpu().numpy().astype("uint8")
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_resized = mask_resized * 255
        
        foreground = cv2.bitwise_and(frame, frame, mask=mask_resized)
        output_path = os.path.join(self.output_dir, f"object_{index}.png")
        cv2.imwrite(output_path, foreground)
        
        return output_path

    def _save_detection_frame(self, frame, result):
        """Save the frame where the high-confidence detection was found"""
        try:
            annotated_frame = result.plot()
            output_path = os.path.join(self.output_dir, f"detection_frame_{int(time.time())}.jpg")
            cv2.imwrite(output_path, annotated_frame)
        except Exception as e:
            print(f"Error saving detection frame: {str(e)}")

    def _extract_text(self, image_path):
        """Extract text from image using EasyOCR"""
        try:
            results = self.reader.readtext(image_path)
            return " ".join([res[1] for res in results])
        except Exception as e:
            print(f"Text extraction error: {str(e)}")
            return ""

    def _display_frame(self, frame, result):
        """Save processed frame with annotations instead of displaying"""
        try:
            annotated_frame = result.plot()
            if platform.system() == "Darwin":  # Save instead of displaying on macOS
                output_path = os.path.join(self.output_dir, f"processed_frame_{int(time.time())}.jpg")
                cv2.imwrite(output_path, annotated_frame)
            else:
                cv2.imshow("Object Detection", annotated_frame)
                cv2.waitKey(1)
        except Exception as e:
            print(f"Error saving/displaying frame: {str(e)}")

    def save_results(self, objects_data, output_file="vision_output.json"):
        """Save processing results to file"""
        try:
            # Ensure we only save one result
            if objects_data and len(objects_data) > 1:
                objects_data = [objects_data[0]]
                
            with open(output_file, 'w') as f:
                json.dump(objects_data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False