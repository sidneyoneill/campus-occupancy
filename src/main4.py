import os
import cv2
import json
import time
import numpy as np
from collections import defaultdict

# Import the person detection module
from person_detector import PersonDetector
from ultralytics import YOLO

# -----------------------------
# Configuration Settings
# -----------------------------
CONFIG = {
    # Input/Output Paths
    "video_path": "data/videos/video_1.MOV",
    "output_path": "output/combined_occupancy_6.MOV",
    "roi_path": "annotations/chair_locations_2.json",
    "desk_roi_path": "annotations/annotations_only_desk.json",
    "baseline_image_path": "data/images/base/frame_1334.jpg",  # Empty reference frame for desk detection
    "json_output_path": "output/detection_results_6.json",  # New path for JSON output
    
    # Processing Parameters
    "resize_factor": 0.5,          # To speed up processing
    "iou_threshold": 0.15,         # IoU threshold for person overlap with chair

    # Chair Detection Parameters
    "chair_keyframe_interval": 100,  # Do new chair detection every N frames
    "chair_model": "yolov8m_chair_cpu",  # Model for chair detection
    "chair_confidence": 0.4,         # Confidence threshold for chair detection
    
    # Person Detection Cycle Parameters
    "person_cycle_interval": 100,    # Every 100 frames start a new detection cycle
    "person_cycle_length": 10,       # Run person detection for 10 consecutive frames in the cycle
    "person_model": "yolov8m",       # Model for person detection
    "person_confidence": 0.1,        # Confidence threshold for person detection
    "occupancy_required_ratio": 0.3, # If occupancy evidence in >= 30% of detection frames, mark chair occupied

    # Desk Detection Parameters
    "desk_keyframe_interval": 10,    # Do desk detection every N frames
    "diff_threshold": 30,            # Threshold for pixel difference in desk detection
    "proportion_threshold": 0.05,    # Threshold for proportion of changed pixels in desk detection
    
    # Debugging
    "debug_mode": True,              # Enable debug output and visualizations
}

# -----------------------------
# Helper Classes and Functions
# -----------------------------

def debug_print(*args, **kwargs):
    """Print debug messages if debug mode is enabled."""
    if CONFIG.get("debug_mode", False):
        print(*args, **kwargs)

class ChairDetector:
    """
    A simplified YOLO-based chair detector.
    """
    def __init__(self, model=CONFIG["chair_model"], confidence=CONFIG["chair_confidence"]):
        if model.endswith('.pt'):
            model_path = model
        else:
            model_path = os.path.join("models", f"{model}.pt")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at {model_path}. Please ensure the chair model is available.")
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_chairs(self, frame):
        """
        Detect chairs in the given frame.
        Returns a list of dictionaries with keys "bbox" and "confidence".
        Bbox format: [x1, y1, x2, y2]
        """
        results = self.model(frame, verbose=False)
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                if conf >= self.confidence:
                    detections.append({"bbox": bbox, "confidence": conf})
        return detections

class DeskDetector:
    """
    A desk occupancy detector using background subtraction with radial masking.
    """
    def __init__(self, baseline_img, annotations, diff_threshold=CONFIG["diff_threshold"], 
                 proportion_threshold=CONFIG["proportion_threshold"]):
        self.baseline_images = self.create_baseline_images(baseline_img, annotations)
        self.annotations = annotations
        self.diff_threshold = diff_threshold
        self.proportion_threshold = proportion_threshold

    def create_baseline_images(self, empty_frame, annotations):
        """
        Create baseline images by cropping the desk ROI from an empty reference frame.
        Returns a dictionary mapping seat_id -> baseline ROI.
        """
        baseline_images = {}
        frame_height, frame_width = empty_frame.shape[:2]
        debug_print(f"Baseline image dimensions: {frame_width}x{frame_height}")
        
        for seat_id, rois in annotations.items():
            if "desk" in rois:
                x, y, width, height = rois["desk"]
                # Ensure ROI is within frame boundaries
                x = max(0, min(x, frame_width-1))
                y = max(0, min(y, frame_height-1))
                width = min(width, frame_width - x)
                height = min(height, frame_height - y)
                
                if width > 0 and height > 0:
                    roi = empty_frame[y:y+height, x:x+width]
                    baseline_images[seat_id] = roi
                    debug_print(f"Created baseline for desk {seat_id}: {roi.shape}")
                else:
                    print(f"Warning: Invalid ROI dimensions for desk {seat_id}")
        
        return baseline_images

    def create_gaussian_mask(self, shape, sigma_factor=0.5):
        """
        Create a Gaussian weighting mask for an ROI.
        The center is weighted more heavily.
        """
        h, w = shape
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = (h - 1) / 2, (w - 1) / 2
        sigma_y = sigma_factor * h
        sigma_x = sigma_factor * w
        mask = np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                        ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
        return mask / np.max(mask)

    def is_desk_occupied(self, current_frame, seat_id):
        """
        Determine occupancy using a Gaussian-weighted proportion of changed pixels.
        Returns a tuple (occupied (bool), confidence (float)).
        """
        # Check if we have a baseline for this seat_id
        if seat_id not in self.baseline_images:
            debug_print(f"Warning: No baseline image for desk {seat_id}")
            return False, 0.0
            
        # Get the desk ROI from the annotations
        if "desk" not in self.annotations[seat_id]:
            debug_print(f"Warning: No desk annotation for {seat_id}")
            return False, 0.0
            
        x, y, width, height = self.annotations[seat_id]["desk"]
        frame_height, frame_width = current_frame.shape[:2]
        
        # Ensure ROI is within frame boundaries
        x = max(0, min(x, frame_width-1))
        y = max(0, min(y, frame_height-1))
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        
        if width <= 0 or height <= 0:
            debug_print(f"Warning: Invalid ROI dimensions for desk {seat_id}")
            return False, 0.0
            
        current_roi = current_frame[y:y+height, x:x+width]
        baseline_roi = self.baseline_images[seat_id]
        
        if current_roi.shape != baseline_roi.shape:
            debug_print(f"ROI shape mismatch for desk {seat_id}: current {current_roi.shape}, baseline {baseline_roi.shape}")
            current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
        
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY) if len(current_roi.shape) == 3 else current_roi
        baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY) if len(baseline_roi.shape) == 3 else baseline_roi
        
        diff = cv2.absdiff(current_gray, baseline_gray)
        gaussian_mask = self.create_gaussian_mask(diff.shape, sigma_factor=0.5)
        weighted_diff = diff * gaussian_mask
        binary_mask = (weighted_diff > self.diff_threshold).astype(np.uint8)
        weighted_proportion = np.sum(binary_mask * gaussian_mask) / np.sum(gaussian_mask)
        confidence = min(100, weighted_proportion * 100)
        occupied = weighted_proportion > self.proportion_threshold
        
        debug_print(f"Desk {seat_id} occupation: {occupied}, confidence: {confidence:.2f}%")
        return occupied, confidence

    def detect_all_desks(self, frame):
        """
        Detect occupancy for all desks in the frame.
        Returns a dictionary mapping seat_id -> (occupied, confidence).
        """
        results = {}
        for seat_id in self.annotations:
            occupied, confidence = self.is_desk_occupied(frame, seat_id)
            results[seat_id] = (occupied, confidence)
        return results
        
    def visualize_desk_difference(self, current_frame, seat_id):
        """
        Create a visualization of the desk difference detection.
        Returns a visualization frame or None if detection couldn't be performed.
        """
        # Check if we have a baseline for this seat_id
        if seat_id not in self.baseline_images:
            return None
            
        # Get the desk ROI from the annotations
        if "desk" not in self.annotations[seat_id]:
            return None
            
        x, y, width, height = self.annotations[seat_id]["desk"]
        frame_height, frame_width = current_frame.shape[:2]
        
        # Ensure ROI is within frame boundaries
        x = max(0, min(x, frame_width-1))
        y = max(0, min(y, frame_height-1))
        width = min(width, frame_width - x)
        height = min(height, frame_height - y)
        
        if width <= 0 or height <= 0:
            return None
            
        current_roi = current_frame[y:y+height, x:x+width]
        baseline_roi = self.baseline_images[seat_id]
        
        if current_roi.shape != baseline_roi.shape:
            current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
        
        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY) if len(current_roi.shape) == 3 else current_roi
        baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY) if len(baseline_roi.shape) == 3 else baseline_roi
        
        diff = cv2.absdiff(current_gray, baseline_gray)
        gaussian_mask = self.create_gaussian_mask(diff.shape, sigma_factor=0.5)
        weighted_diff = diff * gaussian_mask
        binary_mask = (weighted_diff > self.diff_threshold).astype(np.uint8) * 255
        
        # Create visualization
        vis_h, vis_w = current_roi.shape[:2]
        vis_frame = np.zeros((vis_h, vis_w*3, 3), dtype=np.uint8)
        
        # Convert grayscale to color for visualization
        current_color = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)
        baseline_color = cv2.cvtColor(baseline_gray, cv2.COLOR_GRAY2BGR)
        binary_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        
        vis_frame[:, :vis_w, :] = current_color
        vis_frame[:, vis_w:vis_w*2, :] = baseline_color
        vis_frame[:, vis_w*2:, :] = binary_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_frame, "Current", (10, 20), font, 0.5, (255,255,255), 1)
        cv2.putText(vis_frame, "Baseline", (vis_w + 10, 20), font, 0.5, (255,255,255), 1)
        cv2.putText(vis_frame, "Difference", (vis_w*2 + 10, 20), font, 0.5, (255,255,255), 1)
        
        return vis_frame

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in the format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_center(bbox):
    """
    Return the center point (x, y) of a bbox.
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def point_in_bbox(point, bbox):
    """
    Check if a point (x, y) is inside a bbox [x1, y1, x2, y2].
    """
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

def draw_dashed_rectangle(frame, pt1, pt2, color, thickness=2, dash_length=10):
    """
    Draw a dashed rectangle from pt1 to pt2.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw horizontal dashed lines
    for i in range(x1, x2, dash_length * 2):
        start = (i, y1)
        end = (min(i + dash_length, x2), y1)
        cv2.line(frame, start, end, color, thickness)
    for i in range(x1, x2, dash_length * 2):
        start = (i, y2)
        end = (min(i + dash_length, x2), y2)
        cv2.line(frame, start, end, color, thickness)

    # Draw vertical dashed lines
    for i in range(y1, y2, dash_length * 2):
        start = (x1, i)
        end = (x1, min(i + dash_length, y2))
        cv2.line(frame, start, end, color, thickness)
    for i in range(y1, y2, dash_length * 2):
        start = (x2, i)
        end = (x2, min(i + dash_length, y2))
        cv2.line(frame, start, end, color, thickness)

def load_chair_rois(roi_path):
    """
    Load chair ROIs from a JSON file.
    Expects a format like: { "ROI_1": {"chair": [x, y, w, h]}, ... }
    Converts each "chair" ROI to [x, y, x+w, y+h].
    """
    try:
        with open(roi_path, 'r') as f:
            rois = json.load(f)
        converted = {}
        for key, roi in rois.items():
            if "chair" in roi:
                x, y, w, h = roi["chair"]
                # Convert seat IDs to space IDs
                space_id = key.replace("seat", "space") if key.startswith("seat") else key
                converted[space_id] = [x, y, x + w, y + h]
        return converted
    except Exception as e:
        print("Error loading chair ROIs:", e)
        return {}

def load_desk_annotations(annotations_path):
    """
    Load desk ROI annotations from a JSON file.
    """
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
            # Convert seat IDs to space IDs
            converted = {}
            for key, value in data.items():
                space_id = key.replace("seat", "space") if key.startswith("seat") else key
                converted[space_id] = value
            return converted
    except Exception as e:
        print("Error loading desk annotations:", e)
        return {}

def map_seats_to_spaces(chair_rois, desk_annotations):
    """
    Create a mapping between seat/desk IDs and unified space IDs.
    Return a dictionary mapping space_id -> {"chair_id": chair_id, "desk_id": desk_id}
    """
    space_mapping = {}
    
    # First, add all chair IDs
    for chair_id in chair_rois:
        space_mapping[chair_id] = {"chair_id": chair_id, "desk_id": None}
    
    # Then, add or update with desk IDs
    for desk_id in desk_annotations:
        if desk_id in space_mapping:
            space_mapping[desk_id]["desk_id"] = desk_id
        else:
            space_mapping[desk_id] = {"chair_id": None, "desk_id": desk_id}
    
    return space_mapping

# New function to initialize the occupancy timeline structure
def initialize_occupancy_timeline():
    """
    Initialize a data structure to track occupancy changes over time.
    Returns a dictionary matching the ground truth format.
    """
    return {
        "video_name": os.path.basename(CONFIG["video_path"]),
        "fps": 0,  # Will be updated with actual FPS during processing
        "total_frames": 0,  # Will be updated at the end of processing
        "annotations": defaultdict(lambda: {"chair": [], "desk": []})
    }

# New function to record occupancy changes
def record_occupancy_change(timeline, space_id, object_type, frame_num, occupied):
    """
    Record a change in occupancy for a chair or desk.
    
    Args:
        timeline: The occupancy timeline data structure
        space_id: The ID of the space (e.g., "space1")
        object_type: Either "chair" or "desk"
        frame_num: The frame number where the change occurred
        occupied: Boolean indicating if the object is now occupied or not
    """
    # Convert any NumPy types to Python native types
    frame_num = int(frame_num)
    occupied = bool(occupied)  # Converts np.bool_ to Python bool
    
    # Get the list of segments for this space and object type
    segments = timeline["annotations"][space_id][object_type]
    
    # If this is the first change, create a segment starting from frame 1
    if not segments:
        segments.append({
            "start_frame": 1,
            "end_frame": frame_num - 1,
            "occupied": not occupied  # The previous state was the opposite
        })
    
    # Add the new segment starting from this frame
    # The end_frame will be updated when the next change occurs or at the end of processing
    segments.append({
        "start_frame": frame_num,
        "end_frame": frame_num,  # Temporary, will be updated later
        "occupied": occupied
    })

# New function to update the end frame of the current segments
def update_segment_end_frames(timeline, current_frame):
    """
    Update the end frame of all current segments to the current frame.
    """
    for space_id, objects in timeline["annotations"].items():
        for object_type, segments in objects.items():
            if segments:  # If there are any segments
                segments[-1]["end_frame"] = current_frame

# New function to finalize the timeline by removing any invalid segments
def finalize_timeline(timeline, total_frames):
    """
    Finalize the timeline by setting the total frames and 
    removing any segments with the same start and end frame.
    """
    timeline["total_frames"] = total_frames
    
    # Convert defaultdict to regular dict for JSON serialization
    timeline["annotations"] = dict(timeline["annotations"])
    
    # Process each space
    for space_id, objects in timeline["annotations"].items():
        for object_type, segments in objects.items():
            # Remove any segments where start == end (no real duration)
            objects[object_type] = [seg for seg in segments if seg["start_frame"] < seg["end_frame"]]
            
            # Ensure the last segment extends to the end of the video
            if objects[object_type]:
                objects[object_type][-1]["end_frame"] = total_frames

# -----------------------------
# Main Processing Function
# -----------------------------

def process_video():
    # Configuration settings from CONFIG dictionary
    video_path = CONFIG["video_path"]
    output_path = CONFIG["output_path"]
    chair_roi_path = CONFIG["roi_path"]
    desk_roi_path = CONFIG["desk_roi_path"]
    baseline_image_path = CONFIG["baseline_image_path"]
    json_output_path = CONFIG["json_output_path"]  # New JSON output path
    
    # Load parameters for each detection type
    chair_keyframe_interval = CONFIG["chair_keyframe_interval"]
    desk_keyframe_interval = CONFIG["desk_keyframe_interval"]
    iou_threshold = CONFIG["iou_threshold"]
    resize_factor = CONFIG["resize_factor"]

    person_cycle_interval = CONFIG["person_cycle_interval"]
    person_cycle_length = CONFIG["person_cycle_length"]
    occupancy_required_ratio = CONFIG["occupancy_required_ratio"]

    # Initialize video capture to get frame dimensions
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        print("Error opening video for frame size detection.")
        return
    
    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_cap.release()
    
    debug_print(f"Video dimensions: {frame_width}x{frame_height}")

    # Load chair ROIs and desk annotations
    chair_rois = load_chair_rois(chair_roi_path)
    desk_annotations = load_desk_annotations(desk_roi_path)
    
    debug_print(f"Loaded {len(chair_rois)} chair ROIs")
    debug_print(f"Loaded {len(desk_annotations)} desk annotations")
    
    # Create a mapping between seat IDs and space IDs
    space_mapping = map_seats_to_spaces(chair_rois, desk_annotations)
    debug_print(f"Created {len(space_mapping)} space mappings")
    
    # Load baseline image for desk detection
    debug_print(f"Loading baseline image from: {baseline_image_path}")
    baseline_img = cv2.imread(baseline_image_path)
    if baseline_img is None:
        print(f"Error: Could not load baseline image from {baseline_image_path}")
        return
    
    # Resize baseline image to match video dimensions
    baseline_img = cv2.resize(baseline_img, (frame_width, frame_height))
    debug_print(f"Resized baseline image to {frame_width}x{frame_height}")

    # The chair tracker maps an ROI id to the latest detection:
    # { "bbox": [...], "confidence": ..., "occupied": bool }
    chair_tracker = {}

    # Initialize occupancy evidence dictionary for person detections.
    # Structure: { roi_id: {'positive': count, 'total': count} }
    occupancy_evidence = {}

    # Initialize combined occupancy status
    space_occupancy = {space_id: False for space_id in space_mapping}

    # Initialize the occupancy timeline
    occupancy_timeline = initialize_occupancy_timeline()
    
    # Track the previous occupancy state to detect changes
    previous_occupancy = {}

    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    occupancy_timeline["fps"] = fps  # Set the actual FPS in our timeline
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Initialize detectors
    person_detector = PersonDetector(model=CONFIG["person_model"], confidence=CONFIG["person_confidence"])
    chair_detector = ChairDetector(model=CONFIG["chair_model"], confidence=CONFIG["chair_confidence"])
    desk_detector = DeskDetector(baseline_img, desk_annotations)

    frame_count = 0
    desk_vis_windows = {}  # Track visualization windows

    while True:
        # Start timing for this frame
        frame_start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (int(frame_width * resize_factor), int(frame_height * resize_factor)))

        # --- Chair Detection (runs on chair keyframes) ---
        if frame_count == 1 or frame_count % chair_keyframe_interval == 0:
            detections = chair_detector.detect_chairs(resized_frame)
            # For each detection, update the chair tracker if its center lies within any ROI.
            for det in detections:
                bbox = det["bbox"]
                center = get_center(bbox)
                # Scale detection center back to original frame coordinates
                center = (center[0] / resize_factor, center[1] / resize_factor)
                for roi_id, roi_bbox in chair_rois.items():
                    if point_in_bbox(center, roi_bbox):
                        # Always keep the most recent detection for this ROI.
                        scaled_bbox = [coord / resize_factor for coord in bbox]
                        chair_tracker[roi_id] = {"bbox": scaled_bbox,
                                                "confidence": det["confidence"],
                                                "occupied": False}
                        break  # Update one ROI per detection

        # --- Person Detection Cycle ---
        cycle_index = frame_count % person_cycle_interval
        if cycle_index < person_cycle_length:
            # Run person detection only during the active window of the cycle.
            person_detections = person_detector.detect_people(resized_frame)
            # Scale person detections back to original frame size.
            for person in person_detections:
                person["bbox"] = [coord / resize_factor for coord in person["bbox"]]
            # For each tracked chair, accumulate occupancy evidence.
            for roi_id, chair in chair_tracker.items():
                # Initialize evidence structure if needed.
                if roi_id not in occupancy_evidence:
                    occupancy_evidence[roi_id] = {'positive': 0, 'total': 0}
                occupancy_evidence[roi_id]['total'] += 1
                for person in person_detections:
                    if calculate_iou(chair["bbox"], person["bbox"]) > iou_threshold:
                        occupancy_evidence[roi_id]['positive'] += 1
                        break  # Count only once per frame for each chair
                
                # Update chair occupancy status after each frame's evidence
                if occupancy_evidence[roi_id]['total'] > 0:
                    ratio = occupancy_evidence[roi_id]['positive'] / occupancy_evidence[roi_id]['total']
                    chair_tracker[roi_id]["occupied"] = ratio >= occupancy_required_ratio
        else:
            # Not in active detection window.
            person_detections = []

        # At the end of the active detection window, just reset the evidence for next cycle.
        if cycle_index == person_cycle_length - 1:
            # Clear evidence for next cycle.
            occupancy_evidence = {}

        # --- Desk Detection (runs on desk keyframes) ---
        if frame_count == 1 or frame_count % desk_keyframe_interval == 0:
            try:
                desk_results = desk_detector.detect_all_desks(frame)
                
                # Visualize desk detection differences if in debug mode
                if CONFIG.get("debug_mode", False):
                    for seat_id in desk_annotations:
                        diff_vis = desk_detector.visualize_desk_difference(frame, seat_id)
                        if diff_vis is not None:
                            window_name = f"Desk {seat_id} Difference"
                            cv2.imshow(window_name, diff_vis)
                            desk_vis_windows[window_name] = True
                
                # --- Combine Chair and Desk occupancy results ---
                for space_id, ids in space_mapping.items():
                    chair_id = ids["chair_id"]
                    desk_id = ids["desk_id"]
                    
                    # Default to not occupied
                    chair_occupied = False
                    desk_occupied = False
                    
                    # Check chair occupancy
                    if chair_id and chair_id in chair_tracker:
                        chair_occupied = chair_tracker[chair_id]["occupied"]
                    
                    # Check desk occupancy
                    if desk_id and desk_id in desk_results:
                        desk_occupied, desk_confidence = desk_results[desk_id]
                        if desk_occupied:
                            debug_print(f"Frame {frame_count}: Desk {desk_id} detected as occupied with confidence {desk_confidence:.2f}%")
                    
                    # A space is occupied if either chair or desk is occupied
                    space_occupancy[space_id] = chair_occupied or desk_occupied
                    
                    # Check if this is the first frame
                    if frame_count == 1:
                        # Initialize previous_occupancy for this space
                        previous_occupancy[space_id] = {
                            "chair": chair_occupied,
                            "desk": desk_occupied
                        }
                        
                        # Record initial state in the timeline
                        normalized_space_id = space_id.replace("space", "") if space_id.startswith("space") else space_id
                        record_occupancy_change(occupancy_timeline, f"space{normalized_space_id}", "chair", 1, chair_occupied)
                        record_occupancy_change(occupancy_timeline, f"space{normalized_space_id}", "desk", 1, desk_occupied)
                    else:
                        # Check for changes in chair occupancy
                        if chair_id and previous_occupancy.get(space_id, {}).get("chair") != chair_occupied:
                            normalized_space_id = space_id.replace("space", "") if space_id.startswith("space") else space_id
                            record_occupancy_change(occupancy_timeline, f"space{normalized_space_id}", "chair", frame_count, chair_occupied)
                            previous_occupancy[space_id]["chair"] = chair_occupied
                        
                        # Check for changes in desk occupancy
                        if desk_id and previous_occupancy.get(space_id, {}).get("desk") != desk_occupied:
                            normalized_space_id = space_id.replace("space", "") if space_id.startswith("space") else space_id
                            record_occupancy_change(occupancy_timeline, f"space{normalized_space_id}", "desk", frame_count, desk_occupied)
                            previous_occupancy[space_id]["desk"] = desk_occupied
                
                # Update the end frame of all current segments
                update_segment_end_frames(occupancy_timeline, frame_count)
                
            except Exception as e:
                print(f"Error in desk detection: {e}")
                import traceback
                traceback.print_exc()
                # Continue with just chair detection if desk detection fails

        # --- Visualization ---
        vis_frame = frame.copy()
        
        # Draw each chair ROI as a dashed blue rectangle
        for roi_id, roi_bbox in chair_rois.items():
            pt1 = (int(roi_bbox[0]), int(roi_bbox[1]))
            pt2 = (int(roi_bbox[2]), int(roi_bbox[3]))
            draw_dashed_rectangle(vis_frame, pt1, pt2, (255, 0, 0), thickness=2, dash_length=10)
            cv2.putText(vis_frame, f"Chair ROI: {roi_id}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw each desk ROI as a dashed green rectangle
        for desk_id, rois in desk_annotations.items():
            if "desk" in rois:
                x, y, width, height = rois["desk"]
                pt1 = (int(x), int(y))
                pt2 = (int(x + width), int(y + height))
                # Change color based on occupancy
                desk_occupied = False
                for space_id, ids in space_mapping.items():
                    if ids["desk_id"] == desk_id and space_occupancy[space_id]:
                        desk_occupied = True
                        break
                color = (0, 0, 255) if desk_occupied else (0, 255, 0)  # Red if occupied, green if empty
                draw_dashed_rectangle(vis_frame, pt1, pt2, color, thickness=2, dash_length=10)
                status = "Occupied" if desk_occupied else "Empty"
                cv2.putText(vis_frame, f"Desk ROI: {desk_id} ({status})", (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracked chairs as solid rectangles along with occupancy status.
        for roi_id, chair in chair_tracker.items():
            bbox = chair["bbox"]
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            # Red if occupied, green if empty
            color = (0, 0, 255) if chair["occupied"] else (0, 255, 0)
            cv2.rectangle(vis_frame, pt1, pt2, color, 2)
            status = "Occupied" if chair["occupied"] else "Empty"
            cv2.putText(vis_frame, f"Chair [{roi_id}]: {status}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Optionally, draw person detections in yellow
        for person in person_detections:
            bbox = person["bbox"]
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(vis_frame, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Person {person['confidence']:.2f}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw combined space occupancy status at the top right of the frame
        y_pos = 30
        for space_id, occupied in space_occupancy.items():
            status = "OCCUPIED" if occupied else "EMPTY"
            color = (0, 0, 255) if occupied else (0, 255, 0)
            # Get text size to align text to right side
            # Simplify the display text to just "Space X"
            display_id = space_id.replace("space", "") if space_id.startswith("space") else space_id
            text = f"Space {display_id}: {status}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            # Position text at top right with some margin
            x_pos = frame_width - text_size[0] - 10
            cv2.putText(vis_frame, text, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 30

        # Calculate and show processing time
        frame_process_time = (time.time() - frame_start_time) * 1000  # convert to ms
        cv2.putText(vis_frame, f"Process time: {frame_process_time:.1f} ms", 
                   (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)

        # Show frame count
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Combined Occupancy Monitoring", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        video_writer.write(vis_frame)

    # Finalize the timeline and write it to a JSON file
    finalize_timeline(occupancy_timeline, frame_count)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    
    # Write the timeline to a JSON file
    with open(json_output_path, 'w') as f:
        json.dump(occupancy_timeline, f, indent=2)
    
    print(f"Occupancy timeline written to {json_output_path}")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
