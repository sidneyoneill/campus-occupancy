import cv2
import json
import os
import numpy as np
from collections import defaultdict

def visualize_ground_truth():
    # Configuration
    video_path = "data/videos/video_1.MOV"  # Update with your video path
    ground_truth_json = "annotations/ground_truth_2.json"
    annotations_json = "annotations/annotations.json"
    output_video_path = "output/ground_truth_visualization.mp4"
    
    # Load ground truth annotations
    with open(ground_truth_json, 'r') as f:
        ground_truth = json.load(f)
    
    # Load ROIs from annotations.json
    with open(annotations_json, 'r') as f:
        annotation_data = json.load(f)
    
    # Convert seat IDs to space IDs for consistency with ground truth
    rois = {}
    for seat_id, objects in annotation_data.items():
        space_id = seat_id.replace("seat", "space")
        rois[space_id] = {
            "chair": objects["chair"],
            "desk": objects["desk"]
        }
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a dictionary for fast lookup of occupancy at any frame
    occupancy_lookup = {}
    for space_id, objects in ground_truth["annotations"].items():
        occupancy_lookup[space_id] = {
            "chair": [],
            "desk": []
        }
        
        # Process chair annotations
        if "chair" in objects:
            for segment in objects["chair"]:
                start_frame = segment["start_frame"]
                end_frame = segment["end_frame"]
                is_occupied = segment["occupied"]
                
                occupancy_lookup[space_id]["chair"].append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "occupied": is_occupied
                })
        
        # Process desk annotations
        if "desk" in objects:
            for segment in objects["desk"]:
                start_frame = segment["start_frame"]
                end_frame = segment["end_frame"]
                is_occupied = segment["occupied"]
                
                occupancy_lookup[space_id]["desk"].append({
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "occupied": is_occupied
                })
    
    # Function to check if a space is occupied at a given frame
    def is_occupied(space_id, object_type, frame_num):
        if space_id not in occupancy_lookup or object_type not in occupancy_lookup[space_id]:
            return False
        
        for segment in occupancy_lookup[space_id][object_type]:
            if segment["start_frame"] <= frame_num <= segment["end_frame"]:
                return segment["occupied"]
        
        return False
    
    # Function to convert bounding box format to polygon points
    def bbox_to_points(bbox):
        x, y, w, h = bbox
        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], np.int32)
    
    # Set up video writer
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize frame navigation variables
    frame_num = 0
    paused = False
    processed_frames = {}  # Cache for processed frames
    
    # Function to process a frame and create visualization
    def process_frame(frame_num):
        if frame_num in processed_frames:
            return processed_frames[frame_num]
            
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None
            
        # Create a visualization overlay on the frame
        for space_id, space_rois in rois.items():
            # Draw chair ROIs
            if "chair" in space_rois:
                chair_occupied = is_occupied(space_id, "chair", frame_num)
                color = (0, 0, 255) if chair_occupied else (0, 255, 0)  # Red if occupied, Green if empty
                
                points = bbox_to_points(space_rois["chair"])
                cv2.polylines(frame, [points], True, color, 2)
                cv2.putText(frame, f"{space_id} chair", 
                           (points[0][0], points[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw desk ROIs
            if "desk" in space_rois:
                desk_occupied = is_occupied(space_id, "desk", frame_num)
                color = (0, 0, 255) if desk_occupied else (0, 255, 0)  # Red if occupied, Green if empty
                
                points = bbox_to_points(space_rois["desk"])
                cv2.polylines(frame, [points], True, color, 2)
                cv2.putText(frame, f"{space_id} desk", 
                           (points[0][0], points[0][1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Cache processed frame (store only a limited number to avoid memory issues)
        if len(processed_frames) > 100:
            # Remove oldest frames
            keys_to_remove = sorted(processed_frames.keys())[:20]
            for key in keys_to_remove:
                del processed_frames[key]
                
        processed_frames[frame_num] = frame.copy()
        return frame
    
    # Main loop for interactive playback
    while True:
        if not paused:
            frame = process_frame(frame_num)
            if frame is None:
                break
                
            # Write frame to output video when not paused
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Ground Truth Visualization', frame)
            
            frame_num += 1
            
            # Show progress
            if frame_num % 100 == 0:
                print(f"Processing frame {frame_num}/{total_frames}")
        
        # Wait for key press with timeout
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        # Handle key presses
        if key == 27:  # ESC key - exit
            break
        elif key == 32:  # Spacebar - pause/play
            paused = not paused
            print("Video", "paused" if paused else "playing")
        elif key in [2555904, 83, 3, 100]:  # Right arrow (different codes) or 'd' key
            frame_num = min(total_frames - 1, frame_num + 1)
            if paused:
                frame = process_frame(frame_num)
                if frame is not None:
                    cv2.imshow('Ground Truth Visualization', frame)
                print(f"Frame: {frame_num}")
        elif key in [2424832, 81, 2, 97]:  # Left arrow or 'a' key
            frame_num = max(0, frame_num - 1)
            if paused:
                frame = process_frame(frame_num)
                if frame is not None:
                    cv2.imshow('Ground Truth Visualization', frame)
                print(f"Frame: {frame_num}")
        elif key in [2490368, 82, 0, 119]:  # Up arrow or 'w' key
            frame_num = min(total_frames - 1, frame_num + 10)
            if paused:
                frame = process_frame(frame_num)
                if frame is not None:
                    cv2.imshow('Ground Truth Visualization', frame)
                print(f"Frame: {frame_num}")
        elif key in [2621440, 84, 1, 122]:  # Down arrow or 'z' key
            frame_num = max(0, frame_num - 10)
            if paused:
                frame = process_frame(frame_num)
                if frame is not None:
                    cv2.imshow('Ground Truth Visualization', frame)
                print(f"Frame: {frame_num}")
        elif key == 110:  # 'n' - mark current position for editing
            print(f"Marked frame {frame_num} for potential edit")
            # You could add code here to save marked frames to a file
            
        # Check if we've reached the end of the video
        if frame_num >= total_frames:
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Visualization saved to {output_video_path}")

if __name__ == "__main__":
    visualize_ground_truth()