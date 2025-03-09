#!/usr/bin/env python3
"""
Person Detector Module

This module provides functionality to detect people in images using the Ultralytics YOLO model.
It can be integrated into an occupancy detection pipeline to count the number of people in a space.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))


class PersonDetector:
    """
    A class for detecting people in images using YOLO models.
    """

    def __init__(self, model: str = "yolov8s", confidence: float = 0.5):
        """
        Initialize the PersonDetector with a YOLO model.

        Args:
            model: YOLO model name ('yolov8s', 'yolov8m', etc.) or path to model file
            confidence: Confidence threshold for detections
        """
        # Determine if input is a model name or path
        if model.endswith('.pt'):
            model_path = model
        else:
            model_path = f"models/{model}.pt"
        
        # Check if model exists, if not, download it
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at {model_path}. Downloading...")
            
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # In COCO dataset, person class has ID 0

    def detect_people(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect people in an image and return their bounding boxes.

        Args:
            image: Input image as a numpy array (BGR format from OpenCV)

        Returns:
            List of dictionaries containing detection information for each person:
            - 'bbox': [x1, y1, x2, y2] (coordinates of the bounding box)
            - 'confidence': Detection confidence score
        """
        # Run the model on the image
        results = self.model(image, verbose=False)[0]
        
        # Filter for person class and confidence threshold
        people_detections = []
        
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            
            # Check if the detection is a person and meets confidence threshold
            if int(class_id) == self.person_class_id and conf >= self.confidence:
                people_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf)
                })
        
        return people_detections

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes for detected people on the image.

        Args:
            image: Input image
            detections: List of detection dictionaries from detect_people()

        Returns:
            Image with bounding boxes drawn
        """
        # Create a copy of the image to draw on
        output_image = image.copy()
        
        # Draw each detection
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Convert to integers for drawing
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle and confidence text
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_image, 
                f"Person: {conf:.2f}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        return output_image

    def count_people(self, image: np.ndarray) -> int:
        """
        Count the number of people in an image.

        Args:
            image: Input image

        Returns:
            Number of people detected
        """
        detections = self.detect_people(image)
        return len(detections)


def main():
    """
    Example usage of the PersonDetector class.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect people in an image using YOLO")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/yolov8s.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, help="Path to save output image (optional)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PersonDetector(model_path=args.model, confidence=args.conf)
    
    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        return
    
    # Detect people
    detections = detector.detect_people(image)
    
    # Print results
    print(f"Detected {len(detections)} people in the image")
    for i, detection in enumerate(detections):
        print(f"Person {i+1}: Confidence = {detection['confidence']:.2f}, BBox = {detection['bbox']}")
    
    # Visualize and save if output path is provided
    if args.output:
        output_image = detector.visualize_detections(image, detections)
        cv2.imwrite(args.output, output_image)
        print(f"Output image saved to {args.output}")


if __name__ == "__main__":
    main() 