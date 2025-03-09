#!/usr/bin/env python3
"""
background_subtraction.py - Script to detect occupancy in ROIs using background subtraction.

This script loads ROI annotations, creates baseline images from an empty reference frame,
and provides a function to detect if ROIs are occupied in new frames.
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

def is_roi_occupied(current_frame, baseline_images, seat_id, roi_type, annotations, threshold=50):
    """
    Determine if an ROI is occupied by comparing it to a baseline image.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        baseline_images (dict): Dictionary of baseline images.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        threshold (int, optional): Threshold for determining occupancy. Defaults to 50.
        
    Returns:
        tuple: (bool, float) - (True if the ROI is occupied, False otherwise, confidence percentage)
    """
    # Get the coordinates of the ROI
    x, y, width, height = annotations[seat_id][roi_type]
    
    # Crop the ROI from the current frame
    current_roi = current_frame[y:y+height, x:x+width]
    
    # Get the baseline image for this ROI
    baseline_roi = baseline_images[(seat_id, roi_type)]
    
    # Ensure the shapes match (in case of any boundary issues)
    if current_roi.shape != baseline_roi.shape:
        print(f"Warning: Shape mismatch for {seat_id} {roi_type}. Resizing.")
        current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
    
    # Convert to grayscale for comparison
    if len(current_roi.shape) == 3:
        current_roi_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    else:
        current_roi_gray = current_roi
        
    if len(baseline_roi.shape) == 3:
        baseline_roi_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY)
    else:
        baseline_roi_gray = baseline_roi
    
    # Calculate the absolute difference between the current ROI and the baseline
    diff = cv2.absdiff(current_roi_gray, baseline_roi_gray)
    
    # Calculate the mean difference
    mean_diff = np.mean(diff)
    
    # Calculate confidence percentage (normalized to 0-100%)
    # We'll use a simple linear mapping where 0 difference is 0% confidence of occupancy
    # and 2*threshold difference is 100% confidence
    confidence = min(100, max(0, (mean_diff / (2 * threshold)) * 100))
    
    # Return True if the mean difference exceeds the threshold, along with confidence
    return mean_diff > threshold, confidence

def visualize_occupancy(frame, annotations, occupied_data):
    """
    Visualize which ROIs are occupied in a frame.
    
    Args:
        frame (numpy.ndarray): Frame to visualize.
        annotations (dict): Dictionary containing ROI annotations.
        occupied_data (dict): Dictionary mapping (seat_id, roi_type) to confidence values.
        
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
            
            # Determine if this ROI is occupied and get confidence
            is_occupied, confidence = False, 0
            if (seat_id, roi_type) in occupied_data:
                is_occupied, confidence = True, occupied_data[(seat_id, roi_type)]
            
            color = colors['occupied'] if is_occupied else colors['empty']
            status = "Occupied" if is_occupied else "Empty"
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x, y), (x + width, y + height), color, 2)
            
            # Create label with confidence percentage
            label = f"{seat_id}: {roi_type} ({status} {confidence:.1f}%)"
            
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
    
    return vis_frame

def main():
    """
    Main function to demonstrate background subtraction for occupancy detection.
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
    # Using a frame from the frames directory as a test frame
    test_frame_path = os.path.join('data', 'images', 'test', 'frame_4988.jpg')
    test_frame = cv2.imread(test_frame_path)
    
    if test_frame is None:
        print(f"Error: Could not load test frame from {test_frame_path}")
        return
    
    # Step 5: Check occupancy for each ROI
    print("Checking occupancy...")
    occupied_data = {}
    
    for seat_id in annotations:
        for roi_type in annotations[seat_id]:
            # Use different thresholds based on ROI type
            if roi_type == "chair":
                threshold = 20  # More sensitive for chairs
            else:  # desk
                threshold = 10  # Less sensitive for desks
            
            # Pass the threshold as a parameter and get both occupancy and confidence
            occupied, confidence = is_roi_occupied(test_frame, baseline_images, seat_id, roi_type, 
                                      annotations, threshold=threshold)
            
            # Print the result
            status = "Occupied" if occupied else "Empty"
            print(f"{seat_id} {roi_type}: {status} (Confidence: {confidence:.1f}%)")
            
            # Add to the dictionary of occupied ROIs if occupied
            if occupied:
                occupied_data[(seat_id, roi_type)] = confidence
    
    # Step 6: Visualize the results
    print("Visualizing results...")
    vis_frame = visualize_occupancy(test_frame, annotations, occupied_data)
    
    # Display the visualization
    cv2.imshow('Occupancy Detection', vis_frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done.")

if __name__ == "__main__":
    main() 