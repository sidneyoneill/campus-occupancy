#!/usr/bin/env python3
"""
Occupancy Monitoring System

This script processes video footage to detect seat occupancy using a combination of
background subtraction and person detection techniques. It visualizes the results
and logs occupancy data with timestamps.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque

import cv2
import numpy as np

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from src.person_detector import PersonDetector
from src.background_subtraction import is_roi_occupied, create_baseline_images


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/occupancy_log_4.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('occupancy_monitor')


# Configuration settings
# You can modify these values directly in the script
CONFIG = {
    # Input/output settings
    'video_path': 'data/videos/video_1.MOV',  # Path to input video
    'annotations_path': 'annotations/annotations.json',  # Path to seat annotations
    'output_path': 'output/processed_video_4.MOV',  # Path to save output video (None to disable)
    'empty_frame_path': 'data/images/base/frame_1334.jpg',  # Path to empty reference frame
    
    # Model settings
    'yolo_model': 'yolov8m',  # YOLO model to use ('yolov8s', 'yolov8m', etc.)
    'person_confidence': 0.2,  # Confidence threshold for person detection
    'iou_threshold': 0.3,  # IoU threshold for person detection overlap with ROIs
    
    # Display and logging settings
    'display': True,  # Whether to display the processed frames
    'log_data': True,  # Whether to log occupancy data
    
    # Processing settings
    'temporal_window': 5,  # Number of frames for temporal smoothing
    'frame_skip': 0,  # Number of frames to skip between processing (0 = process all)
    'show_details': True,  # Whether to show detailed information in visualization
    
    # Background subtraction thresholds
    'chair_bg_threshold': 20,  # Threshold for chair background subtraction
    'desk_bg_threshold': 5,  # Threshold for desk background subtraction
    
    # ROI visualization colors (B,G,R format)
    'colors': {
        'background_occupied': (0, 0, 255),    # Red - occupied via background check
        'person_occupied': (0, 165, 255),      # Orange - occupied via person detection
        'empty': (0, 255, 0),                  # Green - not occupied
        'person_bbox': (0, 255, 255)           # Yellow - person detection box
    }
}

# ROI-specific thresholds
# You can customize these thresholds for each seat and ROI type
ROI_THRESHOLDS = {}

# Default thresholds if not specified in ROI_THRESHOLDS
DEFAULT_THRESHOLDS = {
    'chair': 20,
    'desk': 10
}


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def roi_to_bbox(roi: List[int]) -> List[float]:
    """
    Convert ROI coordinates [x, y, width, height] to bounding box format [x1, y1, x2, y2].
    
    Args:
        roi: ROI coordinates [x, y, width, height]
        
    Returns:
        Bounding box coordinates [x1, y1, x2, y2]
    """
    x, y, width, height = roi
    return [float(x), float(y), float(x + width), float(y + height)]


def fuse_occupancy_results(
    seat_definitions: Dict[str, Dict[str, List[int]]],
    bg_occupancy_results: Dict[str, Dict[str, Tuple[bool, float]]],
    person_detections: List[Dict[str, Any]],
    temporal_history: Optional[Dict[str, deque]] = None,
    temporal_window: int = 5,
    config: Dict[str, Any] = CONFIG
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, deque]]:
    """
    Fuse background subtraction and person detection results to determine seat occupancy.
    
    The fusion logic follows these rules:
    1. If a person is detected overlapping the chair ROI (with sufficient IoU), the seat is 
       immediately marked as occupied with the detection source being 'person_detection'
    2. Otherwise, if either the chair or desk background check indicates occupancy, the seat
       is marked as occupied with the detection source being 'background'
    3. If neither condition is met, the seat is marked as empty
    
    Args:
        seat_definitions: Dictionary mapping seat IDs to ROI coordinates
        bg_occupancy_results: Dictionary with background subtraction results
        person_detections: List of person detection dictionaries
        temporal_history: Optional dictionary to store occupancy history
        temporal_window: Number of frames for temporal smoothing
        config: Configuration dictionary with thresholds and settings
        
    Returns:
        Tuple containing results dictionary and updated temporal history
    """
    # Initialize temporal history if not provided
    if temporal_history is None:
        temporal_history = defaultdict(lambda: deque(maxlen=temporal_window))
    
    # Initialize results dictionary
    results = {}
    
    # Process each seat
    for seat_id, rois in seat_definitions.items():
        # Initialize seat result
        results[seat_id] = {
            'occupied': False,
            'confidence': 0.0,
            'method': 'none',
            'details': {
                'chair': {'occupied': False, 'score': 0.0, 'method': 'none'},
                'desk': {'occupied': False, 'score': 0.0, 'method': 'none'}
            }
        }
        
        # Get chair ROI if available
        chair_roi = rois.get('chair')
        if chair_roi:
            chair_bbox = roi_to_bbox(chair_roi)
            
            # Check for person detections overlapping with chair ROI
            person_detected = False
            max_person_confidence = 0.0
            
            for detection in person_detections:
                if detection['confidence'] < config['person_confidence']:
                    continue
                
                iou = calculate_iou(detection['bbox'], chair_bbox)
                if iou > config['iou_threshold'] and detection['confidence'] > max_person_confidence:
                    person_detected = True
                    max_person_confidence = detection['confidence']
            
            # Get background subtraction result for chair
            bg_chair_occupied, bg_chair_confidence = bg_occupancy_results.get(seat_id, {}).get('chair', (False, 0.0))
            
            # Update chair details
            chair_details = results[seat_id]['details']['chair']
            if person_detected:
                chair_details.update({
                    'occupied': True,
                    'score': max_person_confidence,
                    'method': 'person_detection'
                })
            else:
                chair_details.update({
                    'occupied': bg_chair_occupied,
                    'score': bg_chair_confidence / 100.0,  # Normalize to 0-1
                    'method': 'background'
                })
        
        # Get desk ROI if available
        desk_roi = rois.get('desk')
        if desk_roi:
            # Get background subtraction result for desk
            bg_desk_occupied, bg_desk_confidence = bg_occupancy_results.get(seat_id, {}).get('desk', (False, 0.0))
            
            # Update desk details
            desk_details = results[seat_id]['details']['desk']
            desk_details.update({
                'occupied': bg_desk_occupied,
                'score': bg_desk_confidence / 100.0,  # Normalize to 0-1
                'method': 'background'
            })
        
        # Determine overall seat occupancy using logical OR approach
        chair_details = results[seat_id]['details']['chair']
        desk_details = results[seat_id]['details']['desk']
        
        # If person detected at chair, seat is immediately occupied
        if chair_details.get('method') == 'person_detection' and chair_details['occupied']:
            results[seat_id].update({
                'occupied': True,
                'confidence': chair_details['score'],
                'method': 'person_detection'
            })
        # Otherwise, check if either background check indicates occupancy
        elif (chair_details.get('occupied', False) or desk_details.get('occupied', False)):
            # Use the highest confidence score from the occupied ROIs
            bg_score = max(
                chair_details['score'] if chair_details.get('occupied', False) else 0.0,
                desk_details['score'] if desk_details.get('occupied', False) else 0.0
            )
            results[seat_id].update({
                'occupied': True,
                'confidence': bg_score,
                'method': 'background'
            })
        else:
            # Seat is empty
            results[seat_id].update({
                'occupied': False,
                'confidence': 0.0,
                'method': 'none'
            })
        
        # Apply temporal smoothing
        current_occupancy = results[seat_id]['occupied']
        temporal_history[seat_id].append(current_occupancy)
        smoothed_occupancy = sum(temporal_history[seat_id]) / len(temporal_history[seat_id])
        
        # Update final result with smoothed occupancy
        results[seat_id]['smoothed_confidence'] = smoothed_occupancy
        results[seat_id]['occupied'] = smoothed_occupancy >= 0.5  # Use 0.5 as threshold for temporal smoothing
    
    return results, temporal_history


def visualize_results(
    frame: np.ndarray,
    seat_definitions: Dict[str, Dict[str, List[int]]],
    fusion_results: Dict[str, Dict[str, Any]],
    person_detections: List[Dict[str, Any]],
    show_details: bool = True,
    config: Dict[str, Any] = CONFIG
) -> np.ndarray:
    """
    Visualize detection results on the input frame with separate ROI visualizations.
    
    Color coding:
    - Orange: ROIs and text for seats occupied via person detection
    - Red: ROIs and text for seats occupied via background subtraction
    - Green: ROIs and text for empty seats
    - Yellow: Person detection bounding boxes
    
    Args:
        frame: Input frame
        seat_definitions: Dictionary mapping seat IDs to ROI coordinates
        fusion_results: Dictionary with fusion results
        person_detections: List of person detection dictionaries
        show_details: Whether to show detailed information
        config: Configuration dictionary with visualization settings
        
    Returns:
        Frame with visualization overlays
    """
    # Create a copy of the frame to draw on
    output_frame = frame.copy()
    
    # Draw person detections
    for detection in person_detections:
        bbox = detection['bbox']
        conf = detection['confidence']
        
        # Convert to integers for drawing
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw rectangle and confidence text
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), config['colors']['person_bbox'], 2)
        cv2.putText(
            output_frame, 
            f"Person: {conf:.2f}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            config['colors']['person_bbox'], 
            2
        )
    
    # Draw each seat's status
    for seat_id, result in fusion_results.items():
        # Get ROIs for this seat
        rois = seat_definitions.get(seat_id, {})
        details = result['details']
        
        # Determine overall status color based on detection method
        if result['method'] == 'person_detection':
            status_color = config['colors']['person_occupied']
        elif result['method'] == 'background' and result['occupied']:
            status_color = config['colors']['background_occupied']
        else:
            status_color = config['colors']['empty']
        
        # Draw chair ROI if available
        if 'chair' in rois:
            x, y, w, h = rois['chair']
            chair_details = details['chair']
            
            # Determine chair ROI color based on detection method
            if chair_details['method'] == 'person_detection' and chair_details['occupied']:
                chair_color = config['colors']['person_occupied']
            elif chair_details['method'] == 'background' and chair_details['occupied']:
                chair_color = config['colors']['background_occupied']
            else:
                chair_color = config['colors']['empty']
            
            # Draw chair ROI
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), chair_color, 2)
            
            # Add chair status if showing details
            if show_details:
                status_text = f"Chair: {chair_details['method']}"
                score_text = f"Score: {chair_details['score']:.2f}"
                
                cv2.putText(
                    output_frame,
                    status_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    chair_color,
                    2
                )
                cv2.putText(
                    output_frame,
                    score_text,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    chair_color,
                    2
                )
        
        # Draw desk ROI if available
        if 'desk' in rois:
            x, y, w, h = rois['desk']
            desk_details = details['desk']
            
            # Determine desk ROI color based on occupancy
            if desk_details['occupied']:
                desk_color = config['colors']['background_occupied']
            else:
                desk_color = config['colors']['empty']
            
            # Draw desk ROI
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), desk_color, 2)
            
            # Add desk status if showing details
            if show_details:
                status_text = f"Desk: {desk_details['method']}"
                score_text = f"Score: {desk_details['score']:.2f}"
                
                cv2.putText(
                    output_frame,
                    status_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    desk_color,
                    2
                )
                cv2.putText(
                    output_frame,
                    score_text,
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    desk_color,
                    2
                )
        
        # Add overall seat status
        # Use chair position for label if available, otherwise use desk position
        if 'chair' in rois:
            x, y = rois['chair'][:2]
        elif 'desk' in rois:
            x, y = rois['desk'][:2]
        else:
            continue
        
        # Draw overall status with appropriate color
        cv2.putText(
            output_frame,
            f"Seat {seat_id}: {'Occupied' if result['occupied'] else 'Empty'} ({result['method']})",
            (x, y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        # Add confidence and method if showing details
        if show_details:
            cv2.putText(
                output_frame,
                f"Smoothed conf: {result['smoothed_confidence']:.2f}",
                (x, y - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1
            )
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        output_frame,
        timestamp,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    # Add total occupancy count
    occupied_count = sum(1 for result in fusion_results.values() if result['occupied'])
    total_seats = len(fusion_results)
    cv2.putText(
        output_frame,
        f"Occupancy: {occupied_count}/{total_seats}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return output_frame


def log_occupancy_data(
    fusion_results: Dict[str, Dict[str, Any]],
    timestamp: str,
    log_file: str = 'output/occupancy_data_4.csv'
) -> None:
    """
    Log occupancy data to a CSV file.
    
    Args:
        fusion_results: Dictionary with fusion results
        timestamp: Current timestamp
        log_file: Path to the log file
    """
    # Check if the file exists
    file_exists = os.path.isfile(log_file)
    
    # Open the file in append mode
    with open(log_file, 'a') as f:
        # Write header if the file doesn't exist
        if not file_exists:
            header = 'timestamp,total_seats,occupied_seats'
            for seat_id in fusion_results:
                header += f',seat_{seat_id}'
            f.write(header + '\n')
        
        # Count occupied seats
        occupied_count = sum(1 for result in fusion_results.values() if result['occupied'])
        total_seats = len(fusion_results)
        
        # Write data row
        row = f'{timestamp},{total_seats},{occupied_count}'
        for seat_id, result in sorted(fusion_results.items()):
            row += f',{1 if result["occupied"] else 0}'
        
        f.write(row + '\n')


def process_video() -> None:
    """
    Process a video file to detect seat occupancy using the configuration settings.
    
    The script uses a combination of person detection (YOLO) and background subtraction
    to determine seat occupancy. A seat is considered occupied if either:
    1. A person is detected overlapping the chair ROI (person detection)
    2. Significant changes are detected in the chair or desk ROIs (background subtraction)
    
    The visualization uses different colors to indicate the source of occupancy:
    - Orange: Occupied via person detection
    - Red: Occupied via background subtraction
    - Green: Empty seat
    """
    # Extract configuration settings
    video_path = CONFIG['video_path']
    annotations_path = CONFIG['annotations_path']
    empty_frame_path = CONFIG['empty_frame_path']
    output_path = CONFIG['output_path']
    display = CONFIG['display']
    log_data = CONFIG['log_data']
    temporal_window = CONFIG['temporal_window']
    person_confidence = CONFIG['person_confidence']
    yolo_model = CONFIG['yolo_model']
    frame_skip = CONFIG['frame_skip']
    show_details = CONFIG['show_details']
    
    # Create output directory if needed
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load seat definitions
    with open(annotations_path, 'r') as f:
        seat_definitions = json.load(f)
    
    # Load empty reference frame for background subtraction
    empty_frame = cv2.imread(empty_frame_path)
    if empty_frame is None:
        logger.error(f"Error: Could not load empty reference frame from {empty_frame_path}")
        return
    
    # Create baseline images for background subtraction
    baseline_images = create_baseline_images(empty_frame, seat_definitions)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video at {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output path is provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
    
    # Initialize person detector with specified YOLO model
    detector = PersonDetector(model=yolo_model, confidence=person_confidence)
    
    # Initialize temporal history for smoothing
    temporal_history = None
    
    # Initialize frame counter and processing time tracking
    frame_count = 0
    processing_times = []
    
    logger.info(f"Starting video processing: {video_path}")
    logger.info(f"Using YOLO model: {yolo_model}")
    logger.info(f"Seat definitions loaded from: {annotations_path}")
    logger.info(f"Using empty reference frame: {empty_frame_path}")
    
    # Process video frames
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video reached")
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Skip frames if requested
        if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
            continue
        
        # Start timing
        start_time = time.time()
        
        # Detect people using specified YOLO model
        person_detections = detector.detect_people(frame)
        
        # Apply background subtraction for each ROI
        bg_occupancy_results = {}
        for seat_id, rois in seat_definitions.items():
            bg_occupancy_results[seat_id] = {}
            
            for roi_type, roi in rois.items():
                # Get the appropriate threshold for this ROI type
                threshold = CONFIG[f'{roi_type}_bg_threshold']
                
                # Apply background subtraction
                occupied, confidence = is_roi_occupied(
                    frame, baseline_images, seat_id, roi_type, seat_definitions, threshold
                )
                
                # Store results
                bg_occupancy_results[seat_id][roi_type] = (occupied, confidence)
        
        # Fuse results using logical OR approach
        fusion_results, temporal_history = fuse_occupancy_results(
            seat_definitions,
            bg_occupancy_results,
            person_detections,
            temporal_history,
            temporal_window=temporal_window,
            config=CONFIG
        )
        
        # End timing
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Visualize results with color-coded occupancy sources
        output_frame = visualize_results(
            frame,
            seat_definitions,
            fusion_results,
            person_detections,
            show_details=show_details,
            config=CONFIG
        )
        
        # Add processing time
        cv2.putText(
            output_frame,
            f"Processing time: {processing_time:.3f}s",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Display the frame
        if display:
            cv2.imshow('Occupancy Monitoring', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User interrupted processing")
                break
        
        # Write to output video if enabled
        if video_writer:
            video_writer.write(output_frame)
        
        # Log occupancy data periodically
        if log_data and frame_count % 30 == 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_occupancy_data(fusion_results, timestamp)
            
            occupied_count = sum(1 for result in fusion_results.values() if result['occupied'])
            total_seats = len(fusion_results)
            logger.info(f"Frame {frame_count}: {occupied_count}/{total_seats} seats occupied")
    
    # Calculate and log performance metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        logger.info(f"Average processing time per frame: {avg_time:.3f}s")
        logger.info(f"Effective FPS: {1/avg_time:.2f}")
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    logger.info("Video processing completed")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Process the video using the configuration settings
    process_video() 