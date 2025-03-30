import cv2
import json
import os
import numpy as np
from collections import defaultdict

def create_ground_truth_annotations():
    # Configuration
    video_path = "data/videos/video_1.MOV"  # Update with your video path
    output_json = "annotations/ground_truth.json"
    
    # Load space/chair/desk information
    # This should match your existing JSON files for consistency
    roi_path = "annotations/chair_locations_2.json"
    desk_roi_path = "annotations/annotations_only_desk.json"
    
    with open(roi_path, 'r') as f:
        chair_data = json.load(f)
    
    with open(desk_roi_path, 'r') as f:
        desk_data = json.load(f)
    
    # Create a list of all spaces (replacing "seat" with "space" as in your code)
    spaces = {}
    for key in chair_data:
        space_id = key.replace("seat", "space") if key.startswith("seat") else key
        spaces[space_id] = {"has_chair": True, "has_desk": False}
    
    for key in desk_data:
        space_id = key.replace("seat", "space") if key.startswith("seat") else key
        if space_id in spaces:
            spaces[space_id]["has_desk"] = True
        else:
            spaces[space_id] = {"has_chair": False, "has_desk": True}
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize annotation structure
    annotations = {
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "annotations": {}
    }
    
    for space_id in spaces:
        annotations["annotations"][space_id] = {}
        if spaces[space_id]["has_chair"]:
            annotations["annotations"][space_id]["chair"] = []
        if spaces[space_id]["has_desk"]:
            annotations["annotations"][space_id]["desk"] = []
    
    # Track current state
    current_frame = 0
    occupancy_state = {
        space_id: {
            "chair": {"occupied": False, "start_frame": None},
            "desk": {"occupied": False, "start_frame": None}
        } for space_id in spaces
    }
    
    # Set up control keys
    space_keys = {}
    chair_keys = {}
    desk_keys = {}
    
    # Assign keys to spaces (1-9)
    space_list = sorted(list(spaces.keys()))
    for i, space_id in enumerate(space_list[:9]):  # Limit to 9 spaces for now
        space_keys[ord(str(i+1))] = space_id
        if spaces[space_id]["has_chair"]:
            # Use Q-I for chairs
            chair_keys[ord('q') + i] = space_id
        if spaces[space_id]["has_desk"]:
            # Use A-K for desks
            desk_keys[ord('a') + i] = space_id
    
    # Display instructions
    print("\n========== ANNOTATION CONTROLS ==========")
    print("Space Bar: Play/Pause")
    print("Right Arrow: Forward 1 frame")
    print("Left Arrow: Back 1 frame")
    print("Up Arrow: Forward 10 frames")
    print("Down Arrow: Back 10 frames")
    print("F: Forward 100 frames")
    print("B: Back 100 frames")
    print("S: Save annotations")
    print("ESC: Exit\n")
    
    print("Chair Controls (Toggle Occupancy):")
    for key, space_id in chair_keys.items():
        print(f"{chr(key).upper()}: Chair in {space_id}")
    
    print("\nDesk Controls (Toggle Occupancy):")
    for key, space_id in desk_keys.items():
        print(f"{chr(key).upper()}: Desk in {space_id}")
    print("==========================================\n")
    
    # Function to add state change to annotations
    def record_state_change(space_id, obj_type, end_frame):
        if occupancy_state[space_id][obj_type]["start_frame"] is not None:
            annotations["annotations"][space_id][obj_type].append({
                "start_frame": occupancy_state[space_id][obj_type]["start_frame"],
                "end_frame": end_frame,
                "occupied": occupancy_state[space_id][obj_type]["occupied"]
            })
            # Reset start frame for next segment
            occupancy_state[space_id][obj_type]["start_frame"] = end_frame + 1
    
    # Main annotation loop
    playing = False
    while True:
        # Get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Show frame with annotations overlay
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw frame counter and instructions
        cv2.putText(display_frame, f"Frame: {current_frame}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"{'PLAYING' if playing else 'PAUSED'}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if playing else (0, 0, 255), 2)
        
        # Draw current occupancy states
        y_pos = 100
        for i, space_id in enumerate(space_list):
            if i < 9:  # Only show first 9 spaces
                chair_status = "OCCUPIED" if occupancy_state[space_id]["chair"]["occupied"] else "EMPTY"
                desk_status = "OCCUPIED" if occupancy_state[space_id]["desk"]["occupied"] else "EMPTY"
                
                chair_color = (0, 0, 255) if occupancy_state[space_id]["chair"]["occupied"] else (0, 255, 0)
                desk_color = (0, 0, 255) if occupancy_state[space_id]["desk"]["occupied"] else (0, 255, 0)
                
                if spaces[space_id]["has_chair"]:
                    cv2.putText(display_frame, f"{space_id} Chair: {chair_status} (press {chr(ord('q') + i).upper()})", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, chair_color, 2)
                    y_pos += 30
                
                if spaces[space_id]["has_desk"]:
                    cv2.putText(display_frame, f"{space_id} Desk: {desk_status} (press {chr(ord('a') + i).upper()})", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, desk_color, 2)
                    y_pos += 30
        
        # Display the frame
        cv2.imshow('Annotation Tool', display_frame)
        
        # Playback control
        if playing:
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            current_frame += 1
            if current_frame >= total_frames:
                playing = False
                current_frame = total_frames - 1
        else:
            key = cv2.waitKey(0) & 0xFF
        
        # Process key presses
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Space bar
            playing = not playing
        elif key == ord('s'):  # Save
            # Check if we're not currently using 's' for a chair key
            chair_key_values = list(chair_keys.keys())
            if ord('s') not in chair_key_values:
                # Finalize any open segments
                for space_id in spaces:
                    for obj_type in ["chair", "desk"]:
                        if spaces[space_id][f"has_{obj_type}"]:
                            if occupancy_state[space_id][obj_type]["start_frame"] is not None:
                                record_state_change(space_id, obj_type, current_frame)
                
                # Save to JSON
                os.makedirs(os.path.dirname(output_json), exist_ok=True)
                with open(output_json, 'w') as f:
                    json.dump(annotations, f, indent=2)
                print(f"Annotations saved to {output_json}")
        # For arrow keys, use direct numeric values and common alternatives
        elif key in [2555904, 83, 3, 100]:  # Right arrow (different codes) or 'd' key
            current_frame = min(total_frames - 1, current_frame + 1)
        elif key in [2424832, 81, 2, 97]:  # Left arrow or 'a' key
            current_frame = max(0, current_frame - 1)
        elif key in [2490368, 82, 0, 119]:  # Up arrow or 'w' key
            current_frame = min(total_frames - 1, current_frame + 10)
        elif key in [2621440, 84, 1, 122]:  # Down arrow or 'z' key
            current_frame = max(0, current_frame - 10)
        elif key == ord('f'):  # Forward 100 frames
            # Check if we're not currently using 'f' for a desk key
            desk_key_values = list(desk_keys.keys())
            if ord('f') not in desk_key_values:
                current_frame = min(total_frames - 1, current_frame + 100)
        elif key == ord('b'):  # Back 100 frames
            # Check if we're not currently using 'b' for a desk key
            desk_key_values = list(desk_keys.keys())
            if ord('b') not in desk_key_values:
                current_frame = max(0, current_frame - 100)
        
        # Chair occupancy toggles
        for k, space_id in chair_keys.items():
            if key == k:
                # Record current state
                if occupancy_state[space_id]["chair"]["start_frame"] is not None:
                    record_state_change(space_id, "chair", current_frame - 1)
                
                # Toggle state
                occupancy_state[space_id]["chair"]["occupied"] = not occupancy_state[space_id]["chair"]["occupied"]
                occupancy_state[space_id]["chair"]["start_frame"] = current_frame
                print(f"Frame {current_frame}: {space_id} chair set to {occupancy_state[space_id]['chair']['occupied']}")
        
        # Desk occupancy toggles
        for k, space_id in desk_keys.items():
            if key == k:
                # Record current state
                if occupancy_state[space_id]["desk"]["start_frame"] is not None:
                    record_state_change(space_id, "desk", current_frame - 1)
                
                # Toggle state
                occupancy_state[space_id]["desk"]["occupied"] = not occupancy_state[space_id]["desk"]["occupied"]
                occupancy_state[space_id]["desk"]["start_frame"] = current_frame
                print(f"Frame {current_frame}: {space_id} desk set to {occupancy_state[space_id]['desk']['occupied']}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Save one final time
    for space_id in spaces:
        for obj_type in ["chair", "desk"]:
            if spaces[space_id][f"has_{obj_type}"]:
                if occupancy_state[space_id][obj_type]["start_frame"] is not None:
                    record_state_change(space_id, obj_type, current_frame)
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Final annotations saved to {output_json}")

if __name__ == "__main__":
    create_ground_truth_annotations()