#!/usr/bin/env python3
"""
background_subtraction_mog2_video.py - Improved occupancy detection using MOG2 background subtraction with video streams.

This script enhances chair ROI detection by:
1. Training MOG2 background subtractors on a video sequence of an empty scene
2. Applying the trained subtractors to a test video to detect occupancy
3. Using different parameters for chair and desk ROIs
4. Visualizing the results in real-time

MOG2 works best with a sequence of frames to build the background model, as it can learn
the normal variations in lighting and small movements in the background. This approach is
more robust than using a single static image as the baseline.
"""

import json
import cv2
import os
import numpy as np
import time

# Configuration parameters (hardcoded instead of using command-line arguments)
EMPTY_VIDEO_PATH = 'data/videos/empty_1.MOV'
TEST_VIDEO_PATH = 'data/videos/video_1.MOV'
ANNOTATIONS_PATH = 'annotations/annotations.json'
OCCUPANCY_THRESHOLD = 40
OUTPUT_VIDEO_PATH = 'output/processed_video.MOV'  # Set to None to disable saving

# Set to True to save the output video
SAVE_OUTPUT = True

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
    # Create background subtractor models with different parameters for chairs and desks
    # For chairs: Lower varThreshold (more sensitive) and detect shadows
    chair_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,       # Number of frames to build the background model
        varThreshold=16,   # Lower threshold makes it more sensitive to detect people
        detectShadows=True # Detect shadows to help with chair occupancy
    )
    
    # For desks: Higher varThreshold (less sensitive) and no shadow detection
    desk_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,        # Number of frames to build the background model
        varThreshold=25,    # Higher threshold for desks to reduce false positives
        detectShadows=False # Shadows less important for desk detection
    )
    
    return {
        "chair": chair_subtractor,
        "desk": desk_subtractor
    }

def train_background_subtractors_from_video(video_path, subtractors, annotations):
    """
    Train background subtractor models on a video of an empty scene.
    
    Args:
        video_path (str): Path to the empty scene video.
        subtractors (dict): Dictionary of background subtractor models.
        annotations (dict): Dictionary containing ROI annotations.
        
    Returns:
        bool: True if training was successful, False otherwise.
    """
    print(f"Training background subtractors from video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video contains {frame_count} frames at {fps} FPS")
    
    # Process each frame of the empty scene video
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame to speed up training (adjust as needed)
        if frame_idx % 5 == 0:
            # Extract each ROI and apply it to the corresponding subtractor
            for seat_id, rois in annotations.items():
                for roi_type, coords in rois.items():
                    x, y, width, height = coords
                    roi = frame[y:y+height, x:x+width]
                    
                    # Apply the ROI to the appropriate subtractor
                    # Use a higher learning rate during training (default is -1 which is auto)
                    subtractors[roi_type].apply(roi, learningRate=0.01)
            
            # Display progress
            if frame_idx % 50 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"Training progress: {progress:.1f}% (Frame {frame_idx}/{frame_count})")
                
                # Optional: Display the frame during training
                cv2.imshow('Training on Empty Scene', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        frame_idx += 1
    
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Training completed on {frame_idx} frames")
    return True

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
    # Use zero learning rate during testing to avoid adapting to occupants
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
                label = f"{seat_id}: {roi_type} ({status}, {confidence:.1f}%)"
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
                if roi_section.shape[:2] == colored_mask.shape[:2]:
                    cv2.addWeighted(colored_mask, alpha, roi_section, 1 - alpha, 0, roi_section)
                    vis_frame[y:y+height, x:x+width] = roi_section
    
    # Add method name and timestamp to the image
    method_label = "Method: MOG2 Video"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(vis_frame, method_label, (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(vis_frame, timestamp, (10, 60), font, 0.7, (255, 255, 255), 1)
    
    return vis_frame

def process_test_video(video_path, subtractors, annotations, threshold=40, output_path=None):
    """
    Process a test video to detect occupancy in ROIs.
    
    Args:
        video_path (str): Path to the test video.
        subtractors (dict): Dictionary of trained background subtractor models.
        annotations (dict): Dictionary containing ROI annotations.
        threshold (int, optional): Threshold for determining occupancy. Defaults to 40.
        output_path (str, optional): Path to save the output video. If None, no video is saved.
        
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    print(f"Processing test video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video dimensions: {frame_width}x{frame_height}, {fps} FPS, {frame_count} frames")
    
    # Create video writer if output path is provided
    video_writer = None
    if output_path:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )
        print(f"Saving output video to: {output_path}")
    
    # Process each frame of the test video
    frame_idx = 0
    start_time = time.time()
    
    # Dictionary to track occupancy status over time
    occupancy_history = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check occupancy for each ROI
        occupied_rois = set()
        confidence_values = {}
        masks = {}
        
        for seat_id in annotations:
            for roi_type in annotations[seat_id]:
                # Check if the ROI is occupied using MOG2 method
                occupied, confidence, mask = is_roi_occupied_mog2(
                    frame, subtractors, seat_id, roi_type, annotations, threshold
                )
                
                # Add to the set of occupied ROIs if occupied
                if occupied:
                    occupied_rois.add((seat_id, roi_type))
                
                # Store confidence value and mask
                confidence_values[(seat_id, roi_type)] = confidence
                masks[(seat_id, roi_type)] = mask
                
                # Update occupancy history
                key = (seat_id, roi_type)
                if key not in occupancy_history:
                    occupancy_history[key] = []
                occupancy_history[key].append(occupied)
                # Keep only the last 10 frames for temporal smoothing
                if len(occupancy_history[key]) > 10:
                    occupancy_history[key].pop(0)
        
        # Apply temporal smoothing to reduce flickering
        smoothed_occupied_rois = set()
        for key, history in occupancy_history.items():
            # If more than 50% of recent frames show occupancy, consider it occupied
            if sum(history) / len(history) > 0.5:
                smoothed_occupied_rois.add(key)
        
        # Visualize the results
        vis_frame = visualize_occupancy(frame, annotations, smoothed_occupied_rois, confidence_values, masks)
        
        # Display the frame
        cv2.imshow('Occupancy Detection', vis_frame)
        
        # Write the frame to the output video if enabled
        if video_writer:
            video_writer.write(vis_frame)
        
        # Display progress
        if frame_idx % 30 == 0:
            elapsed_time = time.time() - start_time
            progress = (frame_idx / frame_count) * 100
            fps_processing = frame_idx / elapsed_time if elapsed_time > 0 else 0
            print(f"Processing progress: {progress:.1f}% (Frame {frame_idx}/{frame_count}), {fps_processing:.1f} FPS")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Processing completed on {frame_idx} frames")
    return True

def main():
    """
    Main function to demonstrate MOG2-based background subtraction for occupancy detection using video streams.
    """
    print("=== MOG2 Background Subtraction with Video Streams ===")
    print(f"Empty video: {EMPTY_VIDEO_PATH}")
    print(f"Test video: {TEST_VIDEO_PATH}")
    print(f"Annotations: {ANNOTATIONS_PATH}")
    print(f"Occupancy threshold: {OCCUPANCY_THRESHOLD}")
    if SAVE_OUTPUT:
        print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")
    else:
        print("Output video will not be saved")
    print("=" * 50)
    
    # Step 1: Load annotations
    print("Loading annotations...")
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    # Step 2: Set up background subtractors
    print("Setting up background subtractors...")
    subtractors = setup_background_subtractors()
    
    # Step 3: Train background subtractors from the empty scene video
    if not train_background_subtractors_from_video(EMPTY_VIDEO_PATH, subtractors, annotations):
        print("Error: Failed to train background subtractors")
        return
    
    # Step 4: Process the test video to detect occupancy
    output_path = OUTPUT_VIDEO_PATH if SAVE_OUTPUT else None
    if not process_test_video(TEST_VIDEO_PATH, subtractors, annotations, OCCUPANCY_THRESHOLD, output_path):
        print("Error: Failed to process test video")
        return
    
    print("Done.")

if __name__ == "__main__":
    main() 