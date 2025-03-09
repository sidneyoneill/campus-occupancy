import cv2
import numpy as np
import os
import json
from pathlib import Path

def compute_feature_matching_error(baseline_roi, current_roi):
    """
    Compute feature matching error between two images using ORB features and homography.
    
    Args:
        baseline_roi: Reference image
        current_roi: Current image to compare against baseline
        
    Returns:
        float: Matching error metric (mean reprojection error for inliers or high value if matching fails)
    """
    # Initialize ORB detector with default parameters
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors for both images
    kp1, des1 = orb.detectAndCompute(baseline_roi, None)
    kp2, des2 = orb.detectAndCompute(current_roi, None)

    print(f"Baseline keypoints: {len(kp1)}, Current keypoints: {len(kp2)}")
    
    # Check if enough keypoints were found
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return float('inf')  # Return high error if not enough keypoints
    
    # Create BFMatcher object with Hamming distance (for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Check if we have enough matches for homography
    MIN_MATCHES = 4
    if len(matches) < MIN_MATCHES:
        return float('inf')
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    try:
        # Compute homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return float('inf')
        
        # Get inlier matches
        inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
        
        if not inlier_matches:
            return float('inf')
        
        # Compute reprojection error for inliers
        total_error = 0
        for match in inlier_matches:
            # Get source point
            src_pt = np.float32(kp1[match.queryIdx].pt + (1,)).reshape(-1, 1)
            
            # Get actual destination point
            actual_dst_pt = np.float32(kp2[match.trainIdx].pt)
            
            # Project source point using homography
            projected_pt = np.dot(H, src_pt)
            projected_pt = projected_pt / projected_pt[2]  # Normalize homogeneous coordinates
            projected_pt = projected_pt[:2].flatten()
            
            # Compute error as Euclidean distance
            error = np.linalg.norm(actual_dst_pt - projected_pt)
            total_error += error
        
        # Return mean reprojection error
        return total_error / len(inlier_matches)
        
    except Exception as e:
        print(f"Error computing homography: {str(e)}")
        return float('inf')

def crop_roi(image, roi):
    """
    Crop a region of interest from an image.
    
    Args:
        image: Source image
        roi: Region of interest as [x, y, width, height]
        
    Returns:
        Cropped image
    """
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

def display_roi_with_error(roi, error, seat_id, roi_type, window_name=None):
    """
    Display an ROI with its error score overlaid.
    
    Args:
        roi: ROI image
        error: Error score
        seat_id: Seat identifier
        roi_type: Type of ROI (chair or desk)
        window_name: Optional window name
    """
    # Create a copy of the ROI to avoid modifying the original
    display_img = roi.copy()
    
    # Convert to BGR if grayscale for colored text
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    # Format the error text
    error_text = f"Score: {error:.2f}" if error != float('inf') else "Score: inf"
    
    # Add text to the image
    cv2.putText(
        display_img, 
        f"{seat_id} - {roi_type}", 
        (10, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 0, 255), 
        1
    )
    
    cv2.putText(
        display_img, 
        error_text, 
        (10, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (0, 0, 255), 
        1
    )
    
    # Display the image
    if window_name is None:
        window_name = f"{seat_id}_{roi_type}"
    
    cv2.imshow(window_name, display_img)
    
    return display_img

def visualize_results(image, annotations, results):
    """
    Visualize results by drawing bounding boxes and scores on the image.
    
    Args:
        image: The image to draw on
        annotations: Dictionary of seat annotations
        results: List of dictionaries with seat_id, roi_type, and error
        
    Returns:
        Image with bounding boxes and scores
    """
    # Create a copy of the image to avoid modifying the original
    vis_image = image.copy()
    
    # Convert to BGR if grayscale
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Draw each ROI from the annotations with its score
    for result in results:
        seat_id = result['seat_id']
        roi_type = result['roi_type']
        error = result['error']
        
        # Get coordinates from annotations
        coords = annotations[seat_id][roi_type]
        x, y, width, height = coords
        
        # Draw rectangle (green color)
        color = (0, 255, 0)  # Green in BGR
        cv2.rectangle(vis_image, (x, y), (x + width, y + height), color, 2)
        
        # Format score text
        score_text = f"Score: {error:.2f}" if error != float('inf') else "Score: inf"
        
        # Calculate text position (above the rectangle)
        text_x = x
        text_y = y - 5  # 5 pixels above the rectangle
        
        # Get text size
        text_size = cv2.getTextSize(score_text, font, font_scale, font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(vis_image, 
                     (text_x, text_y - text_size[1]), 
                     (text_x + text_size[0], text_y), 
                     color, -1)
        
        # Draw text
        cv2.putText(vis_image, score_text, (text_x, text_y), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return vis_image

if __name__ == '__main__':
    try:
        # 1. Load baseline and current images
        base_img_path = os.path.join('data', 'images', 'base', 'frame_1334.jpg')
        baseline_img = cv2.imread(base_img_path)
        test_img_path = os.path.join('data', 'images', 'test', 'frame_4988.jpg')
        current_img = cv2.imread(test_img_path)
        
        if baseline_img is None or current_img is None:
            raise ValueError("Failed to load test images")
            
        print(f"Loaded baseline image: {base_img_path}")
        print(f"Loaded current image: {test_img_path}")
        
        # 2. Load seat annotations
        annotations_path = os.path.join('annotations', 'annotations.json')
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            print(f"Loaded annotations from: {annotations_path}")
        except Exception as e:
            raise ValueError(f"Failed to load annotations: {str(e)}")
        
        # 3. Crop ROIs for each seat and ROI type
        roi_crops = {}
        
        for seat_id, seat_data in annotations.items():
            roi_crops[seat_id] = {}
            
            # Process each ROI type (chair and desk)
            for roi_type, roi_coords in seat_data.items():
                try:
                    # Crop ROIs from baseline and current images
                    baseline_roi = crop_roi(baseline_img, roi_coords)
                    current_roi = crop_roi(current_img, roi_coords)
                    
                    # Store cropped ROIs
                    roi_crops[seat_id][roi_type] = {
                        'baseline': baseline_roi,
                        'current': current_roi,
                        'coords': roi_coords
                    }
                    
                    print(f"Cropped {roi_type} ROI for {seat_id}")
                except Exception as e:
                    print(f"Error cropping {roi_type} ROI for {seat_id}: {str(e)}")
        
        # 4. Compute matching error for each ROI pair
        results = []
        display_images = []
        
        for seat_id, seat_rois in roi_crops.items():
            for roi_type, roi_data in seat_rois.items():

                baseline_roi = roi_data['baseline']
                current_roi = roi_data['current']
                
                # Compute matching error
                error = compute_feature_matching_error(baseline_roi, current_roi)
                
                # Print results
                print(f"Seat: {seat_id}, ROI: {roi_type}, Matching Error: {error:.2f}")
                
                # Store results
                results.append({
                    'seat_id': seat_id,
                    'roi_type': roi_type,
                    'error': error
                })
                
                # Display individual ROIs with error scores (optional)
                if roi_type == 'chair':
                    display_img = display_roi_with_error(
                        current_roi, 
                        error, 
                        seat_id, 
                        roi_type
                    )
                    display_images.append(display_img)
        
        # 5. Visualize all results on the full image
        print("Visualizing results on the full image...")
        result_image = visualize_results(current_img, annotations, results)
        cv2.imshow('Feature Matching Results', result_image)
        
        # Wait for a key press to close all windows
        print("Press any key to close all windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error in main program: {str(e)}") 