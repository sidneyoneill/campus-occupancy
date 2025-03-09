#!/usr/bin/env python3
"""
background_subtraction_mog2.py - Improved occupancy detection using MOG2 background subtraction.

This script enhances chair ROI detection by:
1. Using OpenCV's MOG2 background subtractor for more sophisticated detection
2. Applying morphological operations to reduce noise
3. Using different parameters for chair and desk ROIs
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

def setup_background_subtractors():
    """
    Set up background subtractor models for each ROI type.
    
    Returns:
        dict: Dictionary of background subtractor models.
    """
    # Create background subtractor models
    chair_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    desk_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    
    return {
        "chair": chair_subtractor,
        "desk": desk_subtractor
    }

def train_background_subtractors(subtractors, empty_frame, annotations):
    """
    Train background subtractor models on the empty reference frame.
    
    Args:
        subtractors (dict): Dictionary of background subtractor models.
        empty_frame (numpy.ndarray): Empty reference frame.
        annotations (dict): Dictionary containing ROI annotations.
    """
    # Train each subtractor on its respective ROIs
    for seat_id, rois in annotations.items():
        for roi_type, coords in rois.items():
            x, y, width, height = coords
            roi = empty_frame[y:y+height, x:x+width]
            
            # Apply the ROI to the appropriate subtractor multiple times to "learn" it
            for _ in range(10):  # Apply multiple times to build history
                subtractors[roi_type].apply(roi)

def is_roi_occupied_mog2(current_frame, subtractors, seat_id, roi_type, annotations, threshold=40):
    """
    Determine if an ROI is occupied using MOG2 background subtraction.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        subtractors (dict): Dictionary of background subtractor models.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        threshold (int, optional): Threshold for determining occupancy. Defaults to 40.
        
    Returns:
        bool: True if the ROI is occupied, False otherwise.
        float: The calculated foreground percentage for debugging.
        numpy.ndarray: The foreground mask for visualization.
    """
    # Get the coordinates of the ROI
    x, y, width, height = annotations[seat_id][roi_type]
    
    # Crop the ROI from the current frame
    current_roi = current_frame[y:y+height, x:x+width]
    
    # Apply the background subtractor to get the foreground mask
    fg_mask = subtractors[roi_type].apply(current_roi, learningRate=0)
    
    # For chair ROIs, apply morphological operations to reduce noise
    if roi_type == "chair":
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Calculate the percentage of foreground pixels
    fg_pixel_count = np.count_nonzero(fg_mask)
    total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
    fg_percentage = (fg_pixel_count / total_pixels) * 100
    
    # Adjust threshold based on ROI type
    actual_threshold = threshold * 0.8 if roi_type == "chair" else threshold
    
    return fg_percentage > actual_threshold, fg_percentage, fg_mask

def visualize_occupancy(frame, annotations, occupied_rois, confidence_values=None, masks=None):
    """
    Visualize which ROIs are occupied in a frame.
    
    Args:
        frame (numpy.ndarray): Frame to visualize.
        annotations (dict): Dictionary containing ROI annotations.
        occupied_rois (set): Set of (seat_id, roi_type) tuples that are occupied.
        confidence_values (dict, optional): Dictionary of confidence values for each ROI.
        masks (dict, optional): Dictionary of foreground masks for each ROI.
        
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
            
            # If masks are provided, overlay the mask on the ROI
            if masks and (seat_id, roi_type) in masks:
                mask = masks[(seat_id, roi_type)]
                # Create a colored mask for visualization
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                colored_mask[mask > 0] = color
                # Blend the mask with the ROI in the visualization frame
                alpha = 0.3  # Transparency factor
                roi_section = vis_frame[y:y+height, x:x+width]
                cv2.addWeighted(colored_mask, alpha, roi_section, 1 - alpha, 0, roi_section)
                vis_frame[y:y+height, x:x+width] = roi_section
    
    # Add method name to the image
    method_label = "Method: MOG2"
    cv2.putText(vis_frame, method_label, (10, 30), 
               font, 1, (255, 255, 255), 2)
    
    return vis_frame

def main():
    """
    Main function to demonstrate MOG2-based background subtraction for occupancy detection.
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
    
    # Step 3: Set up and train background subtractors
    print("Setting up background subtractors...")
    subtractors = setup_background_subtractors()
    
    print("Training background subtractors...")
    train_background_subtractors(subtractors, empty_frame, annotations)
    
    # Step 4: Load a test frame
    print("Loading test frame...")
    test_frame_path = os.path.join('data', 'images', 'frames', 'frame_1334.jpg')
    test_frame = cv2.imread(test_frame_path)
    
    if test_frame is None:
        print(f"Error: Could not load test frame from {test_frame_path}")
        return
    
    # Step 5: Check occupancy for each ROI
    print("Checking occupancy...")
    occupied_rois = set()
    confidence_values = {}
    masks = {}
    
    for seat_id in annotations:
        for roi_type in annotations[seat_id]:
            # Check if the ROI is occupied using MOG2 method
            occupied, confidence, mask = is_roi_occupied_mog2(
                test_frame, subtractors, seat_id, roi_type, annotations
            )
            
            # Print the result
            status = "Occupied" if occupied else "Empty"
            print(f"{seat_id} {roi_type}: {status} (confidence: {confidence:.2f})")
            
            # Add to the set of occupied ROIs if occupied
            if occupied:
                occupied_rois.add((seat_id, roi_type))
            
            # Store confidence value and mask
            confidence_values[(seat_id, roi_type)] = confidence
            masks[(seat_id, roi_type)] = mask
    
    # Step 6: Visualize the results
    print("Visualizing results...")
    vis_frame = visualize_occupancy(test_frame, annotations, occupied_rois, confidence_values, masks)
    
    # Display the visualization
    cv2.imshow('MOG2-Based Occupancy Detection', vis_frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done.")

if __name__ == "__main__":
    main() 