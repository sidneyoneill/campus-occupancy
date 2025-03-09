#!/usr/bin/env python3
"""
background_subtraction_histogram.py - Improved occupancy detection using histogram comparison.

This script enhances chair ROI detection by:
1. Using histogram comparison instead of pixel-by-pixel difference
2. Applying adaptive thresholding based on ROI type
3. Normalizing histograms for more robust comparison
"""

import json
import cv2
import os
import numpy as np

def load_annotations(annotations_path):
    """
    Load ROI annotations from a JSON file.
    
    Args:
        annotations_path (str): Path to the annotations JSON file.
        
    Returns:
        dict: Dictionary containing ROI annotations.
    """
    with open(annotations_path, 'r') as f:
        return json.load(f)

def create_baseline_images(empty_frame, annotations):
    """
    Create baseline images by cropping ROIs from an empty reference frame.
    
    Args:
        empty_frame (numpy.ndarray): Empty reference frame.
        annotations (dict): Dictionary containing ROI annotations.
        
    Returns:
        dict: Dictionary mapping (seat_id, roi_type) to cropped baseline images.
    """
    baseline_images = {}
    
    for seat_id, rois in annotations.items():
        for roi_type, coords in rois.items():
            x, y, width, height = coords
            # Crop the ROI from the empty frame
            roi = empty_frame[y:y+height, x:x+width]
            # Store the cropped ROI in the baseline images dictionary
            baseline_images[(seat_id, roi_type)] = roi
            
    return baseline_images

def is_roi_occupied_histogram(current_frame, baseline_images, seat_id, roi_type, annotations, 
                             base_threshold=50):
    """
    Determine if an ROI is occupied using histogram comparison.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        baseline_images (dict): Dictionary of baseline images.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        base_threshold (int, optional): Base threshold value. Defaults to 50.
        
    Returns:
        bool: True if the ROI is occupied, False otherwise.
        float: The calculated difference value for debugging.
    """
    # Get the coordinates of the ROI
    x, y, width, height = annotations[seat_id][roi_type]
    
    # Crop the ROI from the current frame
    current_roi = current_frame[y:y+height, x:x+width]
    
    # Get the baseline image for this ROI
    baseline_roi = baseline_images[(seat_id, roi_type)]
    
    # Ensure the shapes match
    if current_roi.shape != baseline_roi.shape:
        current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
    
    # Convert to grayscale
    if len(current_roi.shape) == 3:
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_roi
        
    if len(baseline_roi.shape) == 3:
        baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY)
    else:
        baseline_gray = baseline_roi
    
    # Calculate histograms
    hist_current = cv2.calcHist([current_gray], [0], None, [256], [0, 256])
    hist_baseline = cv2.calcHist([baseline_gray], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_current, hist_current, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_baseline, hist_baseline, 0, 1, cv2.NORM_MINMAX)
    
    # Compare histograms using correlation method (1.0 means perfect match)
    # Lower values indicate more difference (potential occupancy)
    hist_similarity = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CORREL)
    
    # Convert similarity to difference (0 to 100 scale)
    hist_diff = (1 - hist_similarity) * 100
    
    # Adaptive thresholding based on ROI type
    if roi_type == "chair":
        # Lower threshold for chairs (more sensitive)
        threshold = base_threshold * 0.6
    else:
        threshold = base_threshold
    
    return hist_diff > threshold, hist_diff

def visualize_occupancy(frame, annotations, occupied_rois, confidence_values=None):
    """
    Visualize which ROIs are occupied in a frame.
    
    Args:
        frame (numpy.ndarray): Frame to visualize.
        annotations (dict): Dictionary containing ROI annotations.
        occupied_rois (set): Set of (seat_id, roi_type) tuples that are occupied.
        confidence_values (dict, optional): Dictionary of confidence values for each ROI.
        
    Returns:
        numpy.ndarray: Frame with visualized occupancy.
    """
    # Create a copy of the frame to avoid modifying the original
    vis_frame = frame.copy()
    
    # Colors for different states (BGR format)
    colors = {
        'occupied': (0, 0, 255),    # Red for occupied
        'empty': (0, 255, 0)        # Green for empty
    }
    
    # Font settings for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Draw each ROI
    for seat_id, rois in annotations.items():
        for roi_type, coords in rois.items():
            x, y, width, height = coords
            
            # Determine if this ROI is occupied
            is_occupied = (seat_id, roi_type) in occupied_rois
            color = colors['occupied'] if is_occupied else colors['empty']
            status = "Occupied" if is_occupied else "Empty"
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x, y), (x + width, y + height), color, 2)
            
            # Create label with confidence if available
            if confidence_values and (seat_id, roi_type) in confidence_values:
                confidence = confidence_values[(seat_id, roi_type)]
                label = f"{seat_id}: {roi_type} ({status}, {confidence:.1f})"
            else:
                label = f"{seat_id}: {roi_type} ({status})"
            
            # Calculate label position
            label_x = x
            label_y = y - 5
            
            # Draw label background
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(vis_frame, (label_x, label_y - text_size[1]), 
                         (label_x + text_size[0], label_y), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
    
    # Add method name to the image
    method_label = "Method: Histogram"
    cv2.putText(vis_frame, method_label, (10, 30), 
               font, 1, (255, 255, 255), 2)
    
    return vis_frame

def main():
    """
    Main function to demonstrate histogram-based background subtraction for occupancy detection.
    """
    # Step 1: Load annotations
    print("Loading annotations...")
    annotations_path = os.path.join('annotations', 'annotations.json')
    annotations = load_annotations(annotations_path)
    
    # Step 2: Load the empty reference image
    print("Loading empty reference image...")
    empty_frame_path = os.path.join('data', 'images', 'base', 'frame_1334.jpg')
    empty_frame = cv2.imread(empty_frame_path)
    
    if empty_frame is None:
        print(f"Error: Could not load empty reference image from {empty_frame_path}")
        return
    
    # Step 3: Create baseline images for each ROI
    print("Creating baseline images...")
    baseline_images = create_baseline_images(empty_frame, annotations)
    
    # Step 4: Load a test frame
    print("Loading test frame...")
    test_frame_path = os.path.join('data', 'images', 'frames', 'frame_4988.jpg')
    test_frame = cv2.imread(test_frame_path)
    
    if test_frame is None:
        print(f"Error: Could not load test frame from {test_frame_path}")
        return
    
    # Step 5: Check occupancy for each ROI
    print("Checking occupancy...")
    occupied_rois = set()
    confidence_values = {}
    
    for seat_id in annotations:
        for roi_type in annotations[seat_id]:
            # Check if the ROI is occupied using histogram method
            occupied, confidence = is_roi_occupied_histogram(
                test_frame, baseline_images, seat_id, roi_type, annotations
            )
            
            # Print the result
            status = "Occupied" if occupied else "Empty"
            print(f"{seat_id} {roi_type}: {status} (confidence: {confidence:.2f})")
            
            # Add to the set of occupied ROIs if occupied
            if occupied:
                occupied_rois.add((seat_id, roi_type))
            
            # Store confidence value
            confidence_values[(seat_id, roi_type)] = confidence
    
    # Step 6: Visualize the results
    print("Visualizing results...")
    vis_frame = visualize_occupancy(test_frame, annotations, occupied_rois, confidence_values)
    
    # Display the visualization
    cv2.imshow('Histogram-Based Occupancy Detection', vis_frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done.")

if __name__ == "__main__":
    main() 