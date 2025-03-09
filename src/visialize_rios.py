#!/usr/bin/env python3
"""
visualize_rois.py - Script to visualize ROIs from annotations on an image.

This script loads ROI annotations from a JSON file, draws them on an image,
and displays the result.
"""

import json
import cv2
import os

def main():
    """
    Main function to load annotations, draw ROIs on an image, and display the result.
    """
    # Step 1: Load annotations from JSON file
    print("Loading annotations...")
    annotations_path = os.path.join('annotations', 'annotations.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Step 2: Load the image using OpenCV
    print("Loading image...")
    image_path = os.path.join('data', 'images', 'frames', 'frame_1334.jpg')
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Step 3: Draw each ROI as a rectangle with label
    print("Drawing ROIs...")
    # Define colors for different ROI types (BGR format)
    colors = {
        'chair': (0, 255, 0),  # Green for chairs
        'desk': (0, 0, 255)    # Red for desks
    }
    
    # Font settings for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Draw each ROI from the annotations
    for seat_id, rois in annotations.items():
        for roi_type, coords in rois.items():
            # Extract coordinates
            x, y, width, height = coords
            
            # Draw rectangle
            color = colors.get(roi_type, (255, 0, 0))  # Default to blue if type not found
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
            
            # Create label with seat ID and ROI type
            label = f"{seat_id}: {roi_type}"
            
            # Calculate label position (above the rectangle)
            label_x = x
            label_y = y - 5  # 5 pixels above the rectangle
            
            # Draw label background
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(image, (label_x, label_y - text_size[1]), 
                         (label_x + text_size[0], label_y), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (label_x, label_y), 
                       font, font_scale, (255, 255, 255), font_thickness)
    
    # Step 4: Display the annotated image
    print("Displaying annotated image...")
    cv2.imshow('ROI Visualization', image)
    
    # Wait for a key press to exit
    print("Press any key to exit...")
    cv2.waitKey(0)
    
    # Step 5: Close the window
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
