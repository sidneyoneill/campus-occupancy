#!/usr/bin/env python3
"""
Occupancy Monitoring with Dynamic Chair Detection

This script processes video footage to detect seat occupancy using a combination of
background subtraction, person detection, and dynamic chair detection.
A YOLO model fine-tuned for chairs is used to detect chairs on-the-fly. The detection
confidence is used to infer occupancy; if the confidence is below a specified threshold,
we assume the chair is occupied (e.g. by an object like a bag).

Desks are processed using the standard annotated ROIs and background subtraction.
Person detection is used to further validate occupancy by checking for overlaps with
chair bounding boxes.
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

# Define a simple IoU function (similar to occupancy_monitor.py)

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

# Define a new ChairDetector class mirroring PersonDetector

from ultralytics import YOLO

class ChairDetector:
    """
    A detector for chairs using a fine-tuned YOLO model.
    """
    def __init__(self, model: str = "yolov8m_chair_cpu", confidence: float = 0.5):
        # Determine model path similar to PersonDetector
        if model.endswith('.pt'):
            model_path = model
        else:
            model_path = os.path.join('models', f"{model}.pt")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at {model_path}. Please ensure the chair model is available.")
        self.model = YOLO(model_path)
        self.confidence = confidence
        # Assuming the model is fine-tuned for chairs, so class filtering may not be needed

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

# Configuration settings
CONFIG = {
    # Input/output settings
    'video_path': 'data/videos/video_1.MOV',
    'annotations_path': 'annotations/annotations.json',
    'output_path': 'output/processed_video_chair_detection_2.MOV',
    'empty_frame_path': 'data/images/base/frame_1334.jpg',
    
    # Model settings
    'yolo_model': 'yolov8m',  # Person detection model
    'person_confidence': 0.2,
    'chair_model': 'yolov8m_chair_cpu',  # Chair detection model
    'chair_detection_confidence': 0.3,  # Minimum confidence for chair detection to be considered
    'chair_confidence_threshold': 0.5,  # Threshold below which a detected chair is considered occupied
    'chair_detection_interval': 10,     # Run chair detection every 10 frames
    'iou_threshold': 0.3,  # IoU threshold for person overlap with chair

    # Display and logging settings
    'display': True,
    'log_data': True,

    # Processing settings
    'temporal_window': 5,
    'frame_skip': 0,
    'show_details': True,

    # Background subtraction thresholds (for desks only)
    'desk_bg_threshold': 10,  
    
    # Visualization colors (B,G,R)
    'colors': {
        'background_occupied': (0, 0, 255),    # Red for occupied via background
        'person_occupied': (0, 165, 255),      # Orange for person detection
        'empty': (0, 255, 0),                  # Green for empty
        'person_bbox': (0, 255, 255),          # Yellow for person bounding box
        'chair_empty': (0, 255, 0),            # Green for empty chair
        'chair_occupied': (0, 0, 255)          # Red for occupied chair
    }
}

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/occupancy_log_chair_detection.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('occupancy_monitor_chair_detection')


def process_video():
    # Extract configuration
    video_path = CONFIG['video_path']
    annotations_path = CONFIG['annotations_path']
    empty_frame_path = CONFIG['empty_frame_path']
    output_path = CONFIG['output_path']
    display = CONFIG['display']
    log_data = CONFIG['log_data']
    frame_skip = CONFIG['frame_skip']
    show_details = CONFIG['show_details']
    iou_threshold = CONFIG['iou_threshold']
    
    # Create output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load seat definitions from annotations
    # We assume annotations contain static ROIs for desks; chairs will be detected dynamically
    with open(annotations_path, 'r') as f:
        seat_definitions = json.load(f)
    
    # Load empty reference frame for background subtraction (for desks)
    empty_frame = cv2.imread(empty_frame_path)
    if empty_frame is None:
        logger.error(f"Error: Could not load empty frame from {empty_frame_path}")
        return
    baseline_images = create_baseline_images(empty_frame, seat_definitions)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video at {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize detectors
    person_detector = PersonDetector(model=CONFIG['yolo_model'], confidence=CONFIG['person_confidence'])
    chair_detector = ChairDetector(model=CONFIG['chair_model'], confidence=CONFIG['chair_detection_confidence'])
    
    frame_count = 0
    processing_times = []
    persistent_chair_detections = []

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
        
        # Run person detection
        person_detections = person_detector.detect_people(frame)
        
        # Run dynamic chair detection at specified intervals
        if frame_count % CONFIG['chair_detection_interval'] == 0:
            new_detections = chair_detector.detect_chairs(frame)
            if new_detections:
                persistent_chair_detections = new_detections
        chair_detections = persistent_chair_detections
        
        # Process desk occupancy using background subtraction
        desk_results = {}
        for seat_id, rois in seat_definitions.items():
            if 'desk' in rois:
                threshold = CONFIG['desk_bg_threshold']
                occupied, confidence = is_roi_occupied(frame, baseline_images, seat_id, 'desk', seat_definitions, threshold)
                desk_results[seat_id] = { 'occupied': occupied, 'confidence': confidence }
        
        # Process chair occupancy using dynamic detections
        chair_results = []
        for detection in chair_detections:
            bbox = detection["bbox"]  # [x1, y1, x2, y2]
            conf = detection["confidence"]
            occupied = False
            method = ""
            person_conf = 0.0
            # Check if any person detection overlaps with this chair bounding box
            for pd in person_detections:
                iou = calculate_iou(bbox, pd["bbox"])
                if iou > iou_threshold and pd["confidence"] > person_conf:
                    occupied = True
                    method = "person_detection"
                    person_conf = pd["confidence"]
            # If no person overlap, use chair detector confidence
            if not occupied:
                if conf < CONFIG['chair_confidence_threshold']:
                    occupied = True
                    method = "chair_detection_low_conf"
                else:
                    occupied = False
                    method = "chair_detection_high_conf"
            chair_results.append({
                'bbox': bbox,
                'confidence': conf,
                'occupied': occupied,
                'method': method
            })
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Visualization
        vis_frame = frame.copy()
        
        # Draw person detections
        for pd in person_detections:
            x1, y1, x2, y2 = map(int, pd["bbox"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), CONFIG['colors']['person_bbox'], 2)
            cv2.putText(vis_frame, f"Person: {pd['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['colors']['person_bbox'], 2)
        
        # Draw desk ROIs from annotations
        for seat_id, rois in seat_definitions.items():
            if 'desk' in rois:
                x, y, w, h = rois['desk']
                desk_occ = desk_results.get(seat_id, {}).get('occupied', False)
                desk_color = CONFIG['colors']['background_occupied'] if desk_occ else CONFIG['colors']['empty']
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), desk_color, 2)
                if show_details:
                    cv2.putText(vis_frame, f"Desk {seat_id}: {desk_occ}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, desk_color, 2)
        
        # Draw dynamic chair detections
        for chair in chair_results:
            x1, y1, x2, y2 = map(int, chair['bbox'])
            color = CONFIG['colors']['chair_occupied'] if chair['occupied'] else CONFIG['colors']['chair_empty']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Chair: {chair['method']} ({chair['confidence']:.2f})"
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay processing time
        cv2.putText(vis_frame, f"Proc time: {processing_time:.3f}s", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display the frame
        if display:
            cv2.imshow('Occupancy Monitoring - Chair Detection', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User interrupted processing")
                break
        
        # Write frame to output video if enabled
        if video_writer:
            video_writer.write(vis_frame)
        
        # Optionally log occupancy data every 30 frames
        if log_data and frame_count % 30 == 0:
            occupied_desks = sum(1 for res in desk_results.values() if res['occupied'])
            occupied_chairs = sum(1 for res in chair_results if res['occupied'])
            total_desks = len(desk_results)
            total_chairs = len(chair_results)
            logger.info(f"Frame {frame_count}: Desks Occupied {occupied_desks}/{total_desks}, Chairs Occupied {occupied_chairs}/{total_chairs}")
    
    # Log performance metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        logger.info(f"Average processing time per frame: {avg_time:.3f}s (FPS: {1/avg_time:.2f})")

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