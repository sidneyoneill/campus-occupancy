#!/usr/bin/env python3
"""
background_subtraction_combined.py - Improved occupancy detection using a combined approach.

This script enhances chair ROI detection by:
1. Using a weighted combination of multiple detection methods
2. Applying different strategies for chair and desk ROIs
3. Providing confidence scores for occupancy detection
"""

import json
import cv2
import os
import numpy as np
import argparse

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

# Basic method (original)
def is_roi_occupied_basic(current_frame, baseline_images, seat_id, roi_type, annotations, threshold=50):
    """
    Original method for ROI occupancy detection.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        baseline_images (dict): Dictionary of baseline images.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        threshold (int, optional): Threshold for determining occupancy. Defaults to 50.
        
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
    
    # Convert to grayscale for comparison
    if len(current_roi.shape) == 3:
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
    else:
        current_gray = current_roi
        
    if len(baseline_roi.shape) == 3:
        baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY)
    else:
        baseline_gray = baseline_roi
    
    # Calculate the absolute difference
    diff = cv2.absdiff(current_gray, baseline_gray)
    mean_diff = np.mean(diff)
    
    return mean_diff > threshold, mean_diff

# HSV method
def is_roi_occupied_hsv(current_frame, baseline_images, seat_id, roi_type, annotations, 
                        threshold=50, chair_expansion=20):
    """
    Enhanced ROI occupancy detection using HSV color space and expanded chair ROIs.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        baseline_images (dict): Dictionary of baseline images.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        threshold (int, optional): Threshold for determining occupancy. Defaults to 50.
        chair_expansion (int, optional): Pixels to expand chair ROI by. Defaults to 20.
        
    Returns:
        bool: True if the ROI is occupied, False otherwise.
        float: The calculated difference value for debugging.
    """
    # Get the coordinates of the ROI
    x, y, width, height = annotations[seat_id][roi_type]
    
    # For chair ROIs, expand the region to capture more of the person
    if roi_type == "chair":
        # Expand in all directions, but ensure we stay within image bounds
        img_height, img_width = current_frame.shape[:2]
        x = max(0, x - chair_expansion)
        y = max(0, y - chair_expansion)
        width = min(img_width - x, width + 2 * chair_expansion)
        height = min(img_height - y, height + 2 * chair_expansion)
    
    # Crop the ROI from the current frame
    current_roi = current_frame[y:y+height, x:x+width]
    
    # Get the baseline image for this ROI
    # For chair ROIs, we need to recrop the baseline with the expanded dimensions
    if roi_type == "chair":
        # Recrop the baseline image with the expanded dimensions
        orig_x, orig_y, orig_width, orig_height = annotations[seat_id][roi_type]
        empty_frame = cv2.imread(os.path.join('data', 'images', 'base', 'base_image_1.jpg'))
        baseline_roi = empty_frame[y:y+height, x:x+width]
    else:
        baseline_roi = baseline_images[(seat_id, roi_type)]
    
    # Ensure the shapes match
    if current_roi.shape != baseline_roi.shape:
        current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
    
    # Convert to HSV color space
    current_hsv = cv2.cvtColor(current_roi, cv2.COLOR_BGR2HSV)
    baseline_hsv = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2HSV)
    
    # Extract the Value channel (brightness)
    current_v = current_hsv[:,:,2]
    baseline_v = baseline_hsv[:,:,2]
    
    # Calculate the absolute difference in the Value channel
    diff = cv2.absdiff(current_v, baseline_v)
    
    # For chair ROIs, focus on the upper part where a person's torso would be
    if roi_type == "chair":
        # Focus on the upper 2/3 of the ROI where a person's torso would be
        upper_region = diff[:int(2*height/3), :]
        mean_diff = np.mean(upper_region)
    else:
        mean_diff = np.mean(diff)
    
    # Use a lower threshold for chairs to increase sensitivity
    actual_threshold = threshold * 0.7 if roi_type == "chair" else threshold
    
    return mean_diff > actual_threshold, mean_diff

# Histogram method
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

# MOG2 method
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

# Combined method
def is_roi_occupied_combined(current_frame, baseline_images, subtractors, seat_id, roi_type, 
                            annotations, base_threshold=50):
    """
    Combined approach for ROI occupancy detection.
    
    Args:
        current_frame (numpy.ndarray): Current frame to check for occupancy.
        baseline_images (dict): Dictionary of baseline images.
        subtractors (dict): Dictionary of background subtractor models.
        seat_id (str): Seat ID of the ROI to check.
        roi_type (str): Type of ROI (chair, desk).
        annotations (dict): Dictionary containing ROI annotations.
        base_threshold (int, optional): Base threshold value. Defaults to 50.
        
    Returns:
        bool: True if the ROI is occupied, False otherwise.
        float: The calculated confidence score.
        numpy.ndarray: The foreground mask for visualization (if available).
    """
    # For desk ROIs, use simple histogram comparison (works well for desks)
    if roi_type == "desk":
        occupied, hist_diff = is_roi_occupied_histogram(
            current_frame, baseline_images, seat_id, roi_type, annotations, base_threshold
        )
        return occupied, hist_diff, None
    
    # For chair ROIs, use a more sophisticated approach
    if roi_type == "chair":
        # Try MOG2 background subtraction
        occupied_mog2, fg_percentage, fg_mask = is_roi_occupied_mog2(
            current_frame, subtractors, seat_id, roi_type, annotations, base_threshold
        )
        
        # Try HSV-based detection
        occupied_hsv, hsv_diff = is_roi_occupied_hsv(
            current_frame, baseline_images, seat_id, roi_type, annotations, base_threshold
        )
        
        # Try histogram comparison
        occupied_hist, hist_diff = is_roi_occupied_histogram(
            current_frame, baseline_images, seat_id, roi_type, annotations, base_threshold
        )
        
        # Weighted voting system
        votes = 0
        if occupied_mog2: votes += 1.5  # MOG2 gets higher weight
        if occupied_hsv: votes += 1.0
        if occupied_hist: votes += 0.5
        
        # Calculate confidence score (0-100)
        confidence = (
            (fg_percentage / base_threshold) * 50 + 
            (hsv_diff / base_threshold) * 30 + 
            (hist_diff / base_threshold) * 20
        )
        
        return votes >= 1.5, confidence, fg_mask
    
    return False, 0, None

def visualize_occupancy(frame, annotations, results):
    """
    Visualize occupancy detection results.
    
    Args:
        frame (numpy.ndarray): Frame to visualize.
        annotations (dict): Dictionary containing ROI annotations.
        results (dict): Dictionary of results for each ROI.
        
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
            
            # Get result for this ROI
            roi_result = results.get((seat_id, roi_type), {})
            is_occupied = roi_result.get('occupied', False)
            confidence = roi_result.get('confidence', 0)
            mask = roi_result.get('mask', None)
            
            color = colors['occupied'] if is_occupied else colors['empty']
            status = "Occupied" if is_occupied else "Empty"
            
            # Draw rectangle
            cv2.rectangle(vis_frame, (x, y), (x + width, y + height), color, 2)
            
            # Create label with confidence
            label = f"{seat_id}: {roi_type} ({status}, {confidence:.1f})"
            
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
            
            # If mask is available, overlay it on the ROI
            if mask is not None:
                # Create a colored mask for visualization
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                colored_mask[mask > 0] = color
                # Blend the mask with the ROI in the visualization frame
                alpha = 0.3  # Transparency factor
                roi_section = vis_frame[y:y+height, x:x+width]
                if roi_section.shape[:2] == colored_mask.shape[:2]:
                    cv2.addWeighted(colored_mask, alpha, roi_section, 1 - alpha, 0, roi_section)
                    vis_frame[y:y+height, x:x+width] = roi_section
    
    # Add method name to the image
    method_label = "Method: Combined"
    cv2.putText(vis_frame, method_label, (10, 30), 
               font, 1, (255, 255, 255), 2)
    
    return vis_frame

def main():
    """
    Main function to demonstrate combined approach for occupancy detection.
    """
    parser = argparse.ArgumentParser(description='Combined background subtraction for occupancy detection.')
    parser.add_argument('--threshold', type=int, default=50,
                        help='Base threshold for occupancy detection')
    args = parser.parse_args()
    
    # Step 1: Load annotations
    print("Loading annotations...")
    annotations_path = os.path.join('annotations', 'annotations.json')
    annotations = load_annotations(annotations_path)
    
    # Step 2: Load the empty reference image
    print("Loading empty reference image...")
    empty_frame_path = os.path.join('data', 'images', 'base', 'base_image_1.jpg')
    empty_frame = cv2.imread(empty_frame_path)
    
    if empty_frame is None:
        print(f"Error: Could not load empty reference image from {empty_frame_path}")
        return
    
    # Step 3: Create baseline images and set up background subtractors
    print("Creating baseline images...")
    baseline_images = create_baseline_images(empty_frame, annotations)
    
    print("Setting up background subtractors...")
    subtractors = setup_background_subtractors()
    
    print("Training background subtractors...")
    train_background_subtractors(subtractors, empty_frame, annotations)
    
    # Step 4: Load a test frame
    print("Loading test frame...")
    test_frame_path = os.path.join('data', 'images', 'frames', 'frame_4988.jpg')
    test_frame = cv2.imread(test_frame_path)
    
    if test_frame is None:
        print(f"Error: Could not load test frame from {test_frame_path}")
        return
    
    # Step 5: Check occupancy for each ROI using the combined approach
    print("Checking occupancy...")
    results = {}
    
    for seat_id in annotations:
        for roi_type in annotations[seat_id]:
            # Check if the ROI is occupied using the combined method
            occupied, confidence, mask = is_roi_occupied_combined(
                test_frame, baseline_images, subtractors, 
                seat_id, roi_type, annotations, args.threshold
            )
            
            # Store the results
            results[(seat_id, roi_type)] = {
                'occupied': occupied,
                'confidence': confidence,
                'mask': mask
            }
            
            # Print the result
            status = "Occupied" if occupied else "Empty"
            print(f"{seat_id} {roi_type}: {status} (confidence: {confidence:.2f})")
    
    # Step 6: Visualize the results
    print("Visualizing results...")
    vis_frame = visualize_occupancy(test_frame, annotations, results)
    
    # Display the visualization
    cv2.imshow('Combined Occupancy Detection', vis_frame)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done.")

if __name__ == "__main__":
    main() 