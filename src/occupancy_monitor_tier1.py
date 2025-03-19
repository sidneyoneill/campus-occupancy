#!/usr/bin/env python3
"""
Occupancy Monitoring - Tier 1 (Lightweight)

A lightweight implementation focusing on minimal computational requirements.
Features:
- Chair detection only on keyframes (every N frames)
- Simple persistence (last known position only)
- Basic temporal smoothing (moving average of occupancy state)
- No background subtraction
- Lower resolution processing

This implementation prioritizes speed and minimal resource usage,
making it suitable for edge devices or real-time processing.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules from the existing codebase
from src.person_detector import PersonDetector
from src.background_subtraction import is_roi_occupied, create_baseline_images
from ultralytics import YOLO

class ChairDetector:
    """
    A detector for chairs using a fine-tuned YOLO model.
    """
    def __init__(self, model: str = "yolov8m_chair_cpu", confidence: float = 0.5):
        # Determine model path
        if model.endswith('.pt'):
            model_path = model
        else:
            model_path = os.path.join('models', f"{model}.pt")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at {model_path}. Please ensure the chair model is available.")
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_chairs(self, frame):
        """
        Detect chairs in the given frame.
        Returns a list of detections with keys: 'bbox' and 'confidence'.
        Bbox format: [x1, y1, x2, y2].
        """
        results = self.model(frame, verbose=False)
        detections = []
        # Process detections from the first result (assuming one image input)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            # Iterate over each detected box
            for box in boxes:
                # box.xyxy is a tensor with coordinates
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                # Only consider detections above the set confidence threshold
                if conf >= self.confidence:
                    detections.append({"bbox": bbox, "confidence": conf})
        return detections

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in the format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

# Configuration settings
CONFIG = {
    # Input/output settings
    'video_path': 'data/videos/video_1.MOV',
    'annotations_path': 'annotations/annotations.json',
    'output_path': 'output/processed_video_tier1.MOV',
    
    # Model settings
    'yolo_model': 'yolov8m',  # Person detection model
    'person_confidence': 0.3,
    'chair_model': 'yolov8m_chair_cpu',  # Chair detection model
    'chair_detection_confidence': 0.4,
    'chair_confidence_threshold': 0.6,  # Threshold below which a detected chair is considered occupied
    'keyframe_interval': 30,  # Only detect chairs every N frames
    'iou_threshold': 0.3,  # IoU threshold for person overlap with chair
    
    # Processing settings
    'temporal_window_size': 3,  # Number of frames for smoothing
    'resize_factor': 0.5,  # Factor to resize input frames (for speed)
    'frame_skip': 1,  # Process every Nth frame (0 = process all)
    
    # Display and logging settings
    'display': True,
    'log_data': True,
    'show_details': True,
    
    # Visualization colors (B,G,R)
    'colors': {
        'empty': (0, 255, 0),                  # Green for empty
        'person_occupied': (0, 165, 255),      # Orange for person detection 
        'object_occupied': (0, 0, 255),        # Red for object occupied
        'person_bbox': (0, 255, 255)           # Yellow for person bounding box
    }
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/occupancy_log_tier1.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('occupancy_monitor_tier1')


class ChairTracker:
    """
    Simple chair tracker that maintains last known positions and occupancy states.
    """
    def __init__(self, temporal_window_size=3):
        self.chairs = {}  # Dictionary to store chair data by ID
        self.next_id = 0  # Counter for assigning chair IDs
        self.temporal_window_size = temporal_window_size
    
    def update(self, chair_detections):
        """
        Update tracker with new chair detections.
        Simply replaces the existing chairs with new detections.
        """
        # For Tier 1, we just reset and store the current detections
        # This is the simplest approach - just remember the last known positions
        self.chairs = {}
        
        for detection in chair_detections:
            chair_id = self.next_id
            self.next_id += 1
            
            # Initialize chair with detection data
            self.chairs[chair_id] = {
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'occupied': False,
                'occupied_by': None,
                'occupancy_history': deque([False] * self.temporal_window_size, maxlen=self.temporal_window_size)
            }
    
    def get_chairs(self):
        """Return the current list of tracked chairs."""
        return self.chairs
    
    def update_occupancy(self, person_detections, iou_threshold):
        """
        Update the occupancy state of chairs based on person detections.
        """
        # Reset occupied_by status
        for chair_id in self.chairs:
            self.chairs[chair_id]['occupied_by'] = None
        
        # Check each chair against person detections
        for chair_id, chair in self.chairs.items():
            chair_bbox = chair['bbox']
            for person in person_detections:
                person_bbox = person['bbox']
                iou = calculate_iou(chair_bbox, person_bbox)
                
                if iou > iou_threshold:
                    # Mark chair as occupied by a person
                    self.chairs[chair_id]['occupied'] = True
                    self.chairs[chair_id]['occupied_by'] = 'person'
                    break
            
            # Update occupancy history
            self.chairs[chair_id]['occupancy_history'].append(self.chairs[chair_id]['occupied'])
            
            # Smooth occupancy using majority vote from history
            occupancy_sum = sum(self.chairs[chair_id]['occupancy_history'])
            self.chairs[chair_id]['occupied'] = occupancy_sum > (self.temporal_window_size / 2)


def process_video():
    # Extract configuration
    video_path = CONFIG['video_path']
    output_path = CONFIG['output_path']
    resize_factor = CONFIG['resize_factor']
    display = CONFIG['display']
    log_data = CONFIG['log_data']
    keyframe_interval = CONFIG['keyframe_interval']
    frame_skip = CONFIG['frame_skip']
    show_details = CONFIG['show_details']
    iou_threshold = CONFIG['iou_threshold']
    temporal_window_size = CONFIG['temporal_window_size']
    
    # Create output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video at {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate resized dimensions for faster processing
    resized_width = int(frame_width * resize_factor)
    resized_height = int(frame_height * resize_factor)
    
    # Initialize video writer if output path provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize detectors
    person_detector = PersonDetector(model=CONFIG['yolo_model'], confidence=CONFIG['person_confidence'])
    chair_detector = ChairDetector(model=CONFIG['chair_model'], confidence=CONFIG['chair_detection_confidence'])
    
    # Initialize chair tracker
    chair_tracker = ChairTracker(temporal_window_size=temporal_window_size)
    
    frame_count = 0
    processing_times = []

    logger.info(f"Starting video processing: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached")
            break
        
        frame_count += 1
        if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
            continue
        
        start_time = time.time()
        
        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        
        # Only run chair detection on keyframes
        if frame_count % keyframe_interval == 0:
            chair_detections = chair_detector.detect_chairs(resized_frame)
            # Update tracker with new chair detections
            chair_tracker.update(chair_detections)
        
        # Run person detection on every processed frame
        person_detections = person_detector.detect_people(resized_frame)
        
        # Scale person detections back to original frame size
        for pd in person_detections:
            pd['bbox'] = [
                pd['bbox'][0] / resize_factor,
                pd['bbox'][1] / resize_factor,
                pd['bbox'][2] / resize_factor,
                pd['bbox'][3] / resize_factor
            ]
        
        # Update chair occupancy based on person detections
        chair_tracker.update_occupancy(person_detections, iou_threshold)
        
        # Get the current state of all chairs
        chairs = chair_tracker.get_chairs()
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Visualization
        vis_frame = frame.copy()
        
        # Draw person detections
        for pd in person_detections:
            x1, y1, x2, y2 = map(int, pd["bbox"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), CONFIG['colors']['person_bbox'], 2)
            if show_details:
                cv2.putText(vis_frame, f"Person: {pd['confidence']:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['colors']['person_bbox'], 2)
        
        # Draw chairs and their occupancy status
        for chair_id, chair in chairs.items():
            # Scale back to original frame size if needed
            x1, y1, x2, y2 = map(int, chair['bbox'])
            # Adjust coordinates if using resized processing
            if resize_factor != 1.0:
                x1 = int(x1 / resize_factor)
                y1 = int(y1 / resize_factor)
                x2 = int(x2 / resize_factor)
                y2 = int(y2 / resize_factor)
            
            # Determine color based on occupancy
            if chair['occupied']:
                if chair['occupied_by'] == 'person':
                    color = CONFIG['colors']['person_occupied']
                    status = "Person"
                else:
                    color = CONFIG['colors']['object_occupied']
                    status = "Occupied"
            else:
                color = CONFIG['colors']['empty']
                status = "Empty"
            
            # Draw chair bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Show details if configured
            if show_details:
                label = f"Chair {chair_id}: {status}"
                cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay processing time and frame info
        cv2.putText(vis_frame, f"Frame: {frame_count} | Proc: {processing_time:.3f}s", 
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display the frame
        if display:
            cv2.imshow('Occupancy Monitoring - Tier 1 (Lightweight)', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User interrupted processing")
                break
        
        # Write frame to output video if enabled
        if video_writer:
            video_writer.write(vis_frame)
        
        # Optionally log occupancy data every 30 frames
        if log_data and frame_count % 30 == 0:
            occupied_chairs = sum(1 for chair in chairs.values() if chair['occupied'])
            total_chairs = len(chairs)
            logger.info(f"Frame {frame_count}: Chairs Occupied {occupied_chairs}/{total_chairs}")
    
    # Log performance metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        logger.info(f"Average processing time per frame: {avg_time:.3f}s (FPS: {1/avg_time:.2f})")
        logger.info(f"Total frames processed: {frame_count}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    logger.info("Video processing completed")


if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    process_video() 