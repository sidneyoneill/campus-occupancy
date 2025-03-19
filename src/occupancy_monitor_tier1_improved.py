#!/usr/bin/env python3
"""
Occupancy Monitoring - Tier 1 Improved (Lightweight with Chair Memory)

A lightweight implementation focusing on minimal computational requirements.
Features:
- Chair detection only on keyframes (every N frames)
- Chair memory to detect when chairs become occluded by people
- Persistent chair tracking across frames
- Basic temporal smoothing (moving average of occupancy state)
- Lower resolution processing

This implementation prioritizes speed and minimal resource usage,
making it suitable for edge devices or real-time processing while
maintaining the ability to detect when a person sits on a chair.
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
    'output_dir': 'output/test_19',  # Base output directory - will be auto-incremented
    'output_video_name': 'processed_video_tier1_improved.MOV',
    
    # Model settings
    'yolo_model': 'yolov8m',  # Person detection model
    'person_confidence': 0.3,
    'chair_model': 'yolov8m_chair_cpu',  # Chair detection model
    'chair_detection_confidence': 0.4,
    'chair_confidence_threshold': 0.6,  # Threshold below which a detected chair is considered occupied
    'keyframe_interval': 30,  # Only detect chairs every N frames
    'iou_threshold': 0.15,  # IoU threshold for person overlap with chair
    'chair_matching_threshold': 0.5,  # IoU threshold to match chairs across frames
    
    # Chair tracking settings
    'max_chair_age': 120,  # Maximum frames to remember a chair position after it disappears
    'min_disappearance_time': 5,  # Minimum frames chair must be gone before checking for person
    'pending_chair_rounds': 2,  # Number of detection rounds to keep a chair in pending state
    
    # Processing settings
    'temporal_window_size': 5,  # Number of frames for smoothing
    'resize_factor': 0.5,  # Factor to resize input frames (for speed)
    'frame_skip': 0,  # Process every Nth frame (0 = process all)
    
    # Display and logging settings
    'display': True,
    'log_data': True,
    'show_details': True,
    
    # ROI settings
    'use_chair_rois': True,
    'chair_rois_path': 'annotations/chair_locations_2.json',
    'roi_matching_threshold': 0.5,
    'roi_exclusivity': True,
    
    # Visualization colors (B,G,R)
    'colors': {
        'empty': (0, 255, 0),                  # Green for empty
        'person_occupied': (0, 165, 255),      # Orange for person detection 
        'occluded': (0, 0, 255),               # Red for occluded chair (likely object)
        'previously_detected': (200, 200, 200), # Grey for chair memory
        'person_bbox': (0, 255, 255),           # Yellow for person bounding box
        'roi': (255, 0, 255)  # Purple for ROI visualization
    }
}

# Function to find the next available test directory
def get_next_test_directory(base_dir='output'):
    """
    Find the next available test directory by incrementing the test number.
    For example, if output/test_5 exists, this will return output/test_6.
    """
    i = 1
    while True:
        test_dir = os.path.join(base_dir, f'test_{i}')
        if not os.path.exists(test_dir):
            return test_dir
        i += 1

# Set up output directory
if 'output_dir' in CONFIG:
    base_output_dir = CONFIG['output_dir']
    # Check if this is a versioned directory request
    if 'test_' in os.path.basename(base_output_dir):
        # If specific test directory is requested, use it
        actual_output_dir = base_output_dir
    else:
        # Otherwise, find the next available test directory
        actual_output_dir = get_next_test_directory(base_output_dir)
    
    # Create the directory if it doesn't exist
    os.makedirs(actual_output_dir, exist_ok=True)
else:
    actual_output_dir = 'output'
    os.makedirs(actual_output_dir, exist_ok=True)

# Update CONFIG with actual output directory
CONFIG['output_dir'] = actual_output_dir
CONFIG['output_path'] = os.path.join(actual_output_dir, CONFIG.get('output_video_name', 'processed_video.MOV'))

# Set up logging
log_file = os.path.join(actual_output_dir, 'occupancy_log_tier1_improved.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('occupancy_monitor_tier1_improved')


class ImprovedChairTracker:
    """
    Chair tracker that maintains chair positions over time and detects when chairs
    are occupied by people. Chairs have just two states: empty or occupied.
    """
    def __init__(self, matching_threshold=0.5, temporal_window_size=5, pending_chair_rounds=1,
                 use_rois=False, chair_rois=None, roi_matching_threshold=0.5, roi_exclusivity=True):
        self.chairs = {}  # All chair positions: both visible and occupied
        self.pending_removal = {}  # Chairs that disappeared but we're keeping for one more round
        self.pending_counters = {}  # Track how many detection rounds a chair has been pending
        self.next_id = 0  # Counter for assigning chair IDs
        self.matching_threshold = matching_threshold
        self.temporal_window_size = temporal_window_size
        self.pending_chair_rounds = pending_chair_rounds
        self.use_rois = use_rois
        self.chair_rois = chair_rois or {}
        self.roi_matching_threshold = roi_matching_threshold
        self.roi_exclusivity = roi_exclusivity
        self.roi_to_chair_map = {}
        
        if self.use_rois:
            self._initialize_roi_chairs()
    
    def _initialize_roi_chairs(self):
        """Initialize chairs based on ROI definitions."""
        for seat_id, rois in self.chair_rois.items():
            if "chair" in rois:
                x, y, w, h = rois["chair"]
                # Only create chair if ROI has valid dimensions
                if w > 0 and h > 0:
                    chair_id = self.next_id
                    self.next_id += 1
                    
                    self.chairs[chair_id] = {
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 1.0,
                        'roi_id': seat_id,
                        'occupied': False,
                        'occupancy_history': deque([False] * self.temporal_window_size, 
                                               maxlen=self.temporal_window_size),
                        'visible': True,
                        'is_roi_chair': True
                    }
                    self.roi_to_chair_map[seat_id] = chair_id
    
    def update(self, chair_detections, resize_factor=1.0):
        """
        Update tracker with new chair detections.
        On detection frames, update our understanding of chair positions.
        """
        # Scale chair detections to original size if needed
        scaled_detections = []
        for detection in chair_detections:
            if resize_factor != 1.0:
                bbox = detection['bbox']
                scaled_bbox = [
                    bbox[0] / resize_factor,
                    bbox[1] / resize_factor,
                    bbox[2] / resize_factor,
                    bbox[3] / resize_factor
                ]
                scaled_detections.append({
                    'bbox': scaled_bbox,
                    'confidence': detection['confidence']
                })
            else:
                scaled_detections.append(detection)
        
        # Call ROI matching to update self.chairs with ROI-related detections.
        if self.use_rois:
            self._match_detections_to_rois(scaled_detections)
        
        # ----- NEW APPROACH TO PRESERVE ROI MAPPINGS -----
        # Save all chairs that already have an ROI association (and those that are occupied)
        roi_chairs = {cid: chair for cid, chair in self.chairs.items() if 'roi_id' in chair}
        occupied_chairs = {cid: chair for cid, chair in self.chairs.items() if chair['occupied']}
        
        # Reset chairs while re-adding ROI and occupied chairs
        self.chairs = {}
        for cid, chair in occupied_chairs.items():
             self.chairs[cid] = chair
        for cid, chair in roi_chairs.items():
             self.chairs[cid] = chair
        # ---------------------------------------------------
        
        # Extract bounding boxes from the new detections for proximity checking
        new_chair_bboxes = [d['bbox'] for d in scaled_detections]
        
        # Determine which non-occupied chairs should be kept.
        chairs_to_keep = {}
        for chair_id, chair in self.chairs.items():
            # For chairs with ROI, let them persist and be updated later.
            if 'roi_id' in chair:
                chairs_to_keep[chair_id] = chair
                continue
            
            should_keep = True
            for new_bbox in new_chair_bboxes:
                if calculate_iou(chair['bbox'], new_bbox) > self.matching_threshold * 0.5:
                    should_keep = False
                    break
            if should_keep:
                chairs_to_keep[chair_id] = chair
        
        # Reset chairs dictionary and then restore the maintained chairs.
        self.chairs = {}
        for chair_id, chair in occupied_chairs.items():
             self.chairs[chair_id] = chair
        for chair_id, chair in chairs_to_keep.items():
             self.chairs[chair_id] = chair
        
        # ----- Process new detections (update or create) -----
        for detection in scaled_detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Skip adding if this detection overlaps an occupied chair
            skip = False
            for _, chair in occupied_chairs.items():
                if calculate_iou(bbox, chair['bbox']) > self.matching_threshold:
                    skip = True
                    break
            if skip:
                continue
            
            # Determine if this detection falls within any ROI.
            roi_id = None
            if self.use_rois:
                for seat_id, rois in self.chair_rois.items():
                    if "chair" not in rois:
                        continue
                    x, y, w, h = rois["chair"]
                    if w <= 0 or h <= 0:
                        continue
                    roi_bbox = [x, y, x + w, y + h]
                    if calculate_iou(bbox, roi_bbox) > self.roi_matching_threshold:
                        roi_id = seat_id
                        break
            
            # NEW LOGIC: In ROI mode, only proceed if the detection falls within an ROI.
            if roi_id is None:
                continue

            # If detection falls within an ROI and there's already a chair in that ROI,
            # update that existing chair.
            if self.use_rois and roi_id and roi_id in self.roi_to_chair_map:
                existing_chair_id = self.roi_to_chair_map[roi_id]
                self.chairs[existing_chair_id] = {
                    'bbox': bbox,
                    'confidence': conf,
                    'roi_id': roi_id,
                    'occupied': False,
                    'occupancy_history': deque([False] * self.temporal_window_size, maxlen=self.temporal_window_size),
                    'visible': True,
                    'is_roi_chair': True
                }
            else:
                # Create a new chair entry
                chair_id = self.next_id
                self.next_id += 1
                chair_data = {
                    'bbox': bbox,
                    'confidence': conf,
                    'occupied': False,
                    'occupancy_history': deque([False] * self.temporal_window_size, maxlen=self.temporal_window_size),
                    'visible': True
                }
                if self.use_rois and roi_id:
                    chair_data['roi_id'] = roi_id
                    chair_data['is_roi_chair'] = True
                    self.roi_to_chair_map[roi_id] = chair_id
                self.chairs[chair_id] = chair_data
        # -----------------------------------------------------
        
        # (Pending chairs handling remains unchanged below)
        chairs_to_remove = []
        new_pending = {}
        for chair_id, chair in self.pending_removal.items():
            if chair['occupied']:
                self.chairs[chair_id] = chair
                chairs_to_remove.append(chair_id)
            else:
                counter = self.pending_counters.get(chair_id, 0) + 1
                self.pending_counters[chair_id] = counter
                if counter >= self.pending_chair_rounds:
                    chairs_to_remove.append(chair_id)
                else:
                    new_pending[chair_id] = chair
        for chair_id in chairs_to_remove:
            if chair_id in self.pending_removal:
                del self.pending_removal[chair_id]
            if chair_id in self.pending_counters:
                del self.pending_counters[chair_id]
        self.pending_removal = new_pending
    
    def _match_detections_to_rois(self, detections):
        """Match detected chairs to defined ROIs."""
        # Track which ROIs have been assigned to avoid duplicates if exclusivity is enabled
        assigned_rois = set()
        detection_matched = [False] * len(detections)
        
        # First pass: match detections to ROIs
        for seat_id, rois in self.chair_rois.items():
            if "chair" not in rois:
                continue
            
            if self.roi_exclusivity and seat_id in assigned_rois:
                continue
                
            # Extract ROI as [x1, y1, x2, y2] for IoU calculation
            x, y, w, h = rois["chair"]
            roi_bbox = [x, y, x + w, y + h]
            
            # Skip invalid ROIs
            if w <= 0 or h <= 0:
                continue
                
            best_match_idx = None
            best_iou = self.roi_matching_threshold  # Minimum threshold to consider a match
            
            # Find best matching detection for this ROI
            for i, detection in enumerate(detections):
                if detection_matched[i] and self.roi_exclusivity:
                    continue  # Skip already matched detections if exclusivity is enabled
                
                iou = calculate_iou(roi_bbox, detection['bbox'])
                if iou > best_iou:
                    best_match_idx = i
                    best_iou = iou
            
            # If a match was found
            if best_match_idx is not None:
                best_match = detections[best_match_idx]
                chair_id = self.roi_to_chair_map.get(seat_id)
                
                if chair_id is not None and chair_id in self.chairs:
                    # Update existing ROI chair
                    self.chairs[chair_id].update({
                        'bbox': best_match['bbox'],
                        'confidence': best_match['confidence'],
                        'visible': True
                    })
                else:
                    # Create new ROI chair
                    chair_id = self.next_id
                    self.next_id += 1
                    self.chairs[chair_id] = {
                        'bbox': best_match['bbox'],
                        'confidence': best_match['confidence'],
                        'roi_id': seat_id,
                        'occupied': False,
                        'occupancy_history': deque([False] * self.temporal_window_size,
                                               maxlen=self.temporal_window_size),
                        'visible': True,
                        'is_roi_chair': True
                    }
                    self.roi_to_chair_map[seat_id] = chair_id
                
                # Mark this detection and ROI as matched
                detection_matched[best_match_idx] = True
                assigned_rois.add(seat_id)

    def update_occupancy(self, person_detections, iou_threshold):
        """
        Update the occupancy state of chairs based on person detections.
        Chairs are either occupied (by a person) or empty.
        """
        # Track all chairs that are occupied by a person
        occupied_chair_ids = set()
        
        # First, reset the current occupancy state for all chairs
        for chair_id in self.chairs:
            self.chairs[chair_id]['currently_occupied'] = False
        
        # Check if a person overlaps with any chair
        for chair_id, chair in self.chairs.items():
            chair_bbox = chair['bbox']
            
            # Check if a person overlaps with this chair
            for person in person_detections:
                person_bbox = person['bbox']
                iou = calculate_iou(chair_bbox, person_bbox)
                
                if iou > iou_threshold:
                    # Mark chair as currently occupied
                    chair['currently_occupied'] = True
                    occupied_chair_ids.add(chair_id)
                    break
            
            # Update occupancy tracking
            if chair.get('currently_occupied', False):
                chair['frames_since_occupied'] = 0
            else:
                chair['frames_since_occupied'] = chair.get('frames_since_occupied', 0) + 1
            
            # Update occupancy history with current state
            chair['occupancy_history'].append(chair.get('currently_occupied', False))
            
            # Calculate smoothed occupancy using weighted average
            recent_occupancy = list(chair['occupancy_history'])
            weights = [i/self.temporal_window_size for i in range(1, self.temporal_window_size+1)]
            weighted_sum = sum(o * w for o, w in zip(recent_occupancy, weights))
            weighted_occupancy = weighted_sum / sum(weights) > 0.5
            
            # Apply smoothed occupancy
            chair['occupied'] = weighted_occupancy
            
            # If chair hasn't been occupied for several consecutive frames, reset its history
            if not chair.get('currently_occupied', False) and chair['frames_since_occupied'] > self.temporal_window_size * 2:
                # No person detected for a while, reset occupancy history
                chair['occupancy_history'] = deque([False] * self.temporal_window_size, 
                                                 maxlen=self.temporal_window_size)
                chair['occupied'] = False
        
        # Also check our pending removal chairs - they might become occupied
        for chair_id, chair in list(self.pending_removal.items()):
            chair['currently_occupied'] = False
            chair_bbox = chair['bbox']
            
            # Check if a person overlaps with this chair
            for person in person_detections:
                person_bbox = person['bbox']
                iou = calculate_iou(chair_bbox, person_bbox)
                
                if iou > iou_threshold:
                    # This chair that disappeared now has a person on it!
                    chair['currently_occupied'] = True
                    chair['frames_since_occupied'] = 0
                    chair['occupancy_history'].append(True)
                    chair['occupied'] = True
                    self.chairs[chair_id] = chair
                    del self.pending_removal[chair_id]
                    break
            
            # Update frames_since_occupied for pending chairs too
            if not chair.get('currently_occupied', False):
                chair['frames_since_occupied'] = chair.get('frames_since_occupied', 0) + 1
                chair['occupancy_history'].append(False)
        
        # After each detection round, mark all chairs as not visible
        # They'll be marked visible again in the next update()
        for chair_id in self.chairs:
            self.chairs[chair_id]['visible'] = False
    
    def get_chairs(self):
        """Return all chairs (both empty and occupied)."""
        # For visualization purposes, include pending chairs
        all_chairs = {**self.chairs, **self.pending_removal}
        return all_chairs

    def get_roi_bboxes(self):
        """Return ROI bounding boxes for visualization."""
        roi_bboxes = {}
        for seat_id, rois in self.chair_rois.items():
            if "chair" in rois:
                x, y, w, h = rois["chair"]
                if w > 0 and h > 0:  # Skip invalid ROIs
                    roi_bboxes[seat_id] = [x, y, x + w, y + h]
        return roi_bboxes


def load_chair_rois(roi_path):
    """Load chair ROIs from JSON file."""
    try:
        with open(roi_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chair ROIs: {e}")
        return {}

def process_video():
    # Extract configuration
    video_path = CONFIG['video_path']
    output_path = CONFIG['output_path']
    output_dir = CONFIG['output_dir']
    resize_factor = CONFIG['resize_factor']
    display = CONFIG['display']
    log_data = CONFIG['log_data']
    keyframe_interval = CONFIG['keyframe_interval']
    frame_skip = CONFIG['frame_skip']
    show_details = CONFIG['show_details']
    iou_threshold = CONFIG['iou_threshold']
    temporal_window_size = CONFIG['temporal_window_size']
    chair_matching_threshold = CONFIG['chair_matching_threshold']
    pending_chair_rounds = CONFIG['pending_chair_rounds']
    use_chair_rois = CONFIG['use_chair_rois']
    
    # Log configuration settings
    logger.info(f"Starting test in directory: {output_dir}")
    logger.info(f"Configuration: {json.dumps({k: v for k, v in CONFIG.items() if k != 'colors'}, indent=2)}")
    
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
    
    # Load chair ROIs if enabled
    chair_rois = None
    if use_chair_rois:
        chair_rois = load_chair_rois(CONFIG['chair_rois_path'])
        logger.info(f"Loaded {len(chair_rois)} chair ROIs")
    
    # Initialize chair tracker with ROI support
    chair_tracker = ImprovedChairTracker(
        matching_threshold=chair_matching_threshold,
        temporal_window_size=temporal_window_size,
        pending_chair_rounds=pending_chair_rounds,
        use_rois=use_chair_rois,
        chair_rois=chair_rois,
        roi_matching_threshold=CONFIG['roi_matching_threshold'],
        roi_exclusivity=CONFIG['roi_exclusivity']
    )
    
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
        
        # Run chair detection on keyframes
        if frame_count % keyframe_interval == 0:
            chair_detections = chair_detector.detect_chairs(resized_frame)
            # Update tracker with new chair detections
            chair_tracker.update(chair_detections, resize_factor)
        
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
        
        # Get all chairs
        chairs = chair_tracker.get_chairs()
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Visualization
        vis_frame = frame.copy()
        
        # Draw ROIs if enabled
        if use_chair_rois:
            roi_bboxes = chair_tracker.get_roi_bboxes()
            for seat_id, bbox in roi_bboxes.items():
                x1, y1, x2, y2 = map(int, bbox)
                # Draw ROI with dashed lines
                dash_length = 15
                roi_color = CONFIG['colors']['roi']
                
                # Draw dashed lines for ROI boundary
                for i in range(0, int((x2-x1)/dash_length)):
                    start_x = x1 + i * dash_length
                    end_x = min(start_x + dash_length//2, x2)
                    cv2.line(vis_frame, (start_x, y1), (end_x, y1), roi_color, 1)
                    cv2.line(vis_frame, (start_x, y2), (end_x, y2), roi_color, 1)
                
                for i in range(0, int((y2-y1)/dash_length)):
                    start_y = y1 + i * dash_length
                    end_y = min(start_y + dash_length//2, y2)
                    cv2.line(vis_frame, (x1, start_y), (x1, end_y), roi_color, 1)
                    cv2.line(vis_frame, (x2, start_y), (x2, end_y), roi_color, 1)
                
                # Add ROI label
                if show_details:
                    cv2.putText(vis_frame, f"ROI: {seat_id}", (x1, y1 - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
        
        # Draw person detections
        for pd in person_detections:
            x1, y1, x2, y2 = map(int, pd["bbox"])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), CONFIG['colors']['person_bbox'], 2)
            if show_details:
                cv2.putText(vis_frame, f"Person: {pd['confidence']:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONFIG['colors']['person_bbox'], 2)
        
        # Draw chairs and their occupancy status
        for chair_id, chair in chairs.items():
            x1, y1, x2, y2 = map(int, chair['bbox'])
            
            # Check if this is a pending chair (scheduled for removal)
            is_pending = chair_id in chair_tracker.pending_removal
            
            # Determine color and status based on occupancy
            if chair['occupied']:
                color = CONFIG['colors']['person_occupied']
                status = "Occupied"
                line_thickness = 2
            else:
                color = CONFIG['colors']['empty']
                status = "Empty"
                if is_pending:
                    status += " (Pending)"
                line_thickness = 2 if not is_pending else 1
            
            # Add ROI info to status if applicable
            if 'roi_id' in chair:
                status += f" [ROI: {chair['roi_id']}]"
            
            # Use dashed lines for chairs that aren't currently visible
            if not chair.get('visible', True) or is_pending:
                # Draw with dashed lines for non-visible or pending chairs
                dash_length = 10
                for i in range(0, int((x2-x1)/dash_length)):
                    start_x = x1 + i * dash_length
                    end_x = min(start_x + dash_length//2, x2)
                    cv2.line(vis_frame, (start_x, y1), (end_x, y1), color, line_thickness)
                    cv2.line(vis_frame, (start_x, y2), (end_x, y2), color, line_thickness)
                
                for i in range(0, int((y2-y1)/dash_length)):
                    start_y = y1 + i * dash_length
                    end_y = min(start_y + dash_length//2, y2)
                    cv2.line(vis_frame, (x1, start_y), (x1, end_y), color, line_thickness)
                    cv2.line(vis_frame, (x2, start_y), (x2, end_y), color, line_thickness)
            else:
                # Draw solid rectangle for visible chairs
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, line_thickness)
            
            # Show chair details
            if show_details:
                label = f"Chair {chair_id}: {status}"
                cv2.putText(vis_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Overlay processing time and frame info
        cv2.putText(vis_frame, f"Frame: {frame_count} | Proc: {processing_time:.3f}s", 
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Display occupancy statistics
        total_chairs = len(chair_tracker.chairs)  # Only count active chairs
        occupied_chairs = sum(1 for chair in chair_tracker.chairs.values() if chair['occupied'])
        pending_chairs = len(chair_tracker.pending_removal)
        
        stats_text = f"Chairs: {total_chairs} | Occupied: {occupied_chairs} | Pending: {pending_chairs}"
        if use_chair_rois:
            roi_chairs = sum(1 for chair in chair_tracker.chairs.values() if 'roi_id' in chair)
            stats_text += f" | In ROIs: {roi_chairs}"
        
        cv2.putText(vis_frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Display the frame
        if display:
            cv2.imshow('Occupancy Monitoring - Tier 1 Improved', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User interrupted processing")
                break
        
        # Write frame to output video if enabled
        if video_writer:
            video_writer.write(vis_frame)
        
        # Optionally log occupancy data every 30 frames
        if log_data and frame_count % 30 == 0:
            visible_chairs = sum(1 for chair in chairs.values() if chair.get('visible', False))
            invisible_chairs = total_chairs - visible_chairs
            
            logger.info(f"Frame {frame_count}: Total Chairs: {total_chairs}, " +
                       f"Visible: {visible_chairs}, Invisible: {invisible_chairs}, Occupied: {occupied_chairs}")
    
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
    
    # Log the start of processing
    logger.info(f"Starting occupancy monitoring in directory: {CONFIG['output_dir']}")
    
    # Save configuration to the output directory for reference
    config_file = os.path.join(CONFIG['output_dir'], 'config.json')
    with open(config_file, 'w') as f:
        json.dump({k: v for k, v in CONFIG.items() if k != 'colors'}, f, indent=2)
    
    # Process the video
    process_video()