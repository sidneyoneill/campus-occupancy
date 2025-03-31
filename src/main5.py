import os
import cv2
import json
import time
import numpy as np
from collections import defaultdict
import traceback  # Added for more detailed error printing

# Import the person detection module
from person_detector import PersonDetector
from ultralytics import YOLO

# -----------------------------
# Configuration Settings
# -----------------------------
CONFIG = {
    # Input/Output Paths
    "video_path": "data/videos/video_1.MOV",
    "output_path": "output/combined_occupancy_main5_v1.MOV", # Updated output name
    "roi_path": "annotations/chair_locations_2.json",
    "desk_roi_path": "annotations/annotations_only_desk.json",
    "baseline_image_path": "data/images/base/frame_1334.jpg",  # Empty reference frame for desk detection
    "json_output_path": "output/detection_results_main5_v1.json", # Updated JSON output name

    # Processing Parameters
    "resize_factor": 0.5,          # To speed up processing
    "iou_threshold": 0.15,         # IoU threshold for person overlap with chair

    # Chair Detection Parameters
    "chair_keyframe_interval": 100,  # Do new chair detection every N frames
    "chair_model": "yolov8m_chair_cpu",  # Model for chair detection
    "chair_confidence": 0.4,         # Confidence threshold for chair detection
    "chair_low_confidence_threshold": 0.2, # NEW: Threshold below which a chair detection implies occupancy (e.g., a bag)

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
                if conf >= self.confidence: # Check against the primary confidence threshold here
                    detections.append({"bbox": bbox, "confidence": conf})
                elif conf > 0: # Also return low confidence detections for the new logic
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
            try:
                current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
            except cv2.error as e:
                 debug_print(f"Error resizing ROI for desk {seat_id}: {e}")
                 return False, 0.0 # Cannot compare if resize fails

        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY) if len(current_roi.shape) == 3 else current_roi
        baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY) if len(baseline_roi.shape) == 3 else baseline_roi

        diff = cv2.absdiff(current_gray, baseline_gray)
        gaussian_mask = self.create_gaussian_mask(diff.shape, sigma_factor=0.5)
        weighted_diff = diff * gaussian_mask
        binary_mask = (weighted_diff > self.diff_threshold).astype(np.uint8)

        # Calculate weighted proportion
        sum_mask = np.sum(gaussian_mask)
        if sum_mask == 0: # Avoid division by zero
            return False, 0.0
        weighted_proportion = np.sum(binary_mask * gaussian_mask) / sum_mask

        # Use proportion directly as confidence (scaled 0-1)
        confidence = min(1.0, weighted_proportion / self.proportion_threshold if self.proportion_threshold > 0 else 1.0) # Scale confidence relative to threshold
        occupied = weighted_proportion > self.proportion_threshold

        # Debug print removed from here, will be printed in main loop

        return occupied, confidence # Return confidence as 0-1 value

    def detect_all_desks(self, frame):
        """
        Detect occupancy for all desks in the frame.
        Returns a dictionary mapping seat_id -> (occupied, confidence).
        """
        results = {}
        for seat_id in self.annotations:
            if "desk" in self.annotations[seat_id]: # Only process if desk ROI exists
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
            try:
                current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
            except cv2.error:
                 return None # Cannot visualize if resize fails

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
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])

    # Draw horizontal dashed lines
    for i in range(x1, x2, dash_length * 2):
        start = (i, y1)
        end = (min(i + dash_length, x2), y1)
        cv2.line(frame, start, end, color, thickness)
        start = (i, y2)
        end = (min(i + dash_length, x2), y2)
        cv2.line(frame, start, end, color, thickness)

    # Draw vertical dashed lines
    for i in range(y1, y2, dash_length * 2):
        start = (x1, i)
        end = (x1, min(i + dash_length, y2))
        cv2.line(frame, start, end, color, thickness)
        start = (x2, i)
        end = (x2, min(i + dash_length, y2))
        cv2.line(frame, start, end, color, thickness)

def load_chair_rois(roi_path):
    """
    Load chair ROIs from a JSON file.
    Expects a format like: { "ROI_1": {"chair": [x, y, w, h]}, ... }
    Converts each "chair" ROI to [x, y, x+w, y+h].
    Uses space IDs directly.
    """
    try:
        with open(roi_path, 'r') as f:
            rois = json.load(f)
        converted = {}
        for key, roi in rois.items():
            if "chair" in roi:
                x, y, w, h = roi["chair"]
                # Use key directly as space_id (assuming it's like "space1")
                converted[key] = [x, y, x + w, y + h]
        return converted
    except Exception as e:
        print(f"Error loading chair ROIs from {roi_path}: {e}")
        return {}

def load_desk_annotations(annotations_path):
    """
    Load desk ROI annotations from a JSON file.
    Uses space IDs directly.
    """
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
            # Use keys directly as space_ids (assuming format is like "space1": {"desk": [...]})
            return data
    except Exception as e:
        print(f"Error loading desk annotations from {annotations_path}: {e}")
        return {}

def map_seats_to_spaces(chair_rois, desk_annotations):
    """
    Create a mapping between chair/desk and unified space IDs.
    Return a dictionary mapping space_id -> {"chair_roi": [x1,y1,x2,y2], "desk_roi": [x,y,w,h]}
    """
    space_mapping = defaultdict(lambda: {"chair_roi": None, "desk_roi": None})

    # Add chair ROIs
    for space_id, roi_bbox in chair_rois.items():
        space_mapping[space_id]["chair_roi"] = roi_bbox

    # Add desk ROIs
    for space_id, annots in desk_annotations.items():
        if "desk" in annots:
            space_mapping[space_id]["desk_roi"] = annots["desk"]

    return dict(space_mapping) # Convert back to regular dict

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
    Record a change in occupancy for a chair or desk. Ensures continuity.

    Args:
        timeline: The occupancy timeline data structure
        space_id: The ID of the space (e.g., "space1")
        object_type: Either "chair" or "desk"
        frame_num: The frame number where the change occurred
        occupied: Boolean indicating if the object is now occupied or not
    """
    frame_num = int(frame_num)
    occupied = bool(occupied)

    segments = timeline["annotations"][space_id][object_type]

    # If it's the first change for this object
    if not segments:
        # If the first detected state is occupied, assume it was empty before
        if occupied:
             segments.append({"start_frame": 1, "end_frame": frame_num -1, "occupied": False})
        # If the first detected state is empty, assume it was occupied before (less likely, but covers all bases)
        else:
             segments.append({"start_frame": 1, "end_frame": frame_num - 1, "occupied": True})
        # Add the first real segment
        segments.append({"start_frame": frame_num, "end_frame": frame_num, "occupied": occupied})
    else:
        # Check if the state actually changed compared to the last segment
        last_segment = segments[-1]
        if last_segment["occupied"] != occupied:
            # Update the end frame of the previous segment
            last_segment["end_frame"] = frame_num - 1
            # Add the new segment
            segments.append({"start_frame": frame_num, "end_frame": frame_num, "occupied": occupied})
        else:
            # If state hasn't changed, just update the end frame of the current segment
             last_segment["end_frame"] = frame_num # Extend the current segment


# New function to update the end frame of the current segments
def update_segment_end_frames(timeline, current_frame):
    """
    Update the end frame of all current segments to the current frame number.
    """
    current_frame = int(current_frame)
    for space_id, objects in timeline["annotations"].items():
        for object_type, segments in objects.items():
            if segments:  # If there are any segments
                # Ensure the last segment's end frame is at least the current frame
                if segments[-1]["end_frame"] < current_frame:
                    segments[-1]["end_frame"] = current_frame


# New function to finalize the timeline by removing any invalid segments
def finalize_timeline(timeline, total_frames):
    """
    Finalize the timeline by setting the total frames and
    removing any segments with start_frame >= end_frame.
    Ensures the last segment extends to total_frames.
    """
    total_frames = int(total_frames)
    timeline["total_frames"] = total_frames

    # Convert defaultdict to regular dict for JSON serialization
    processed_annotations = {}
    for space_id, objects in timeline["annotations"].items():
        processed_objects = {}
        for object_type, segments in objects.items():
            # Filter out invalid segments (start >= end)
            valid_segments = [seg for seg in segments if seg["start_frame"] < seg["end_frame"]]

            # Ensure the last segment extends to the end of the video
            if valid_segments:
                if valid_segments[-1]["end_frame"] < total_frames:
                    valid_segments[-1]["end_frame"] = total_frames
            elif segments: # Handle case where only one segment exists and might need extending
                 if segments[0]["start_frame"] < total_frames:
                      segments[0]["end_frame"] = total_frames
                      valid_segments = segments # Use the original segment if it's now valid
                 else:
                      valid_segments = [] # Still invalid

            processed_objects[object_type] = valid_segments
        processed_annotations[space_id] = processed_objects

    timeline["annotations"] = processed_annotations


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
    json_output_path = CONFIG["json_output_path"]

    # Load parameters
    chair_keyframe_interval = CONFIG["chair_keyframe_interval"]
    chair_confidence_thresh = CONFIG["chair_confidence"]
    chair_low_confidence_threshold = CONFIG["chair_low_confidence_threshold"] # Get the new threshold
    desk_keyframe_interval = CONFIG["desk_keyframe_interval"]
    iou_threshold = CONFIG["iou_threshold"]
    resize_factor = CONFIG["resize_factor"]
    person_cycle_interval = CONFIG["person_cycle_interval"]
    person_cycle_length = CONFIG["person_cycle_length"]
    occupancy_required_ratio = CONFIG["occupancy_required_ratio"]

    # Initialize video capture to get frame dimensions
    temp_cap = cv2.VideoCapture(video_path)
    if not temp_cap.isOpened():
        print(f"Error opening video '{video_path}' for frame size detection.")
        return

    frame_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = temp_cap.get(cv2.CAP_PROP_FPS)
    temp_cap.release()

    debug_print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    if frame_width == 0 or frame_height == 0:
        print("Error: Invalid video dimensions detected.")
        return

    # Load chair ROIs and desk annotations using space IDs
    chair_rois_map = load_chair_rois(chair_roi_path) # space_id -> [x1,y1,x2,y2]
    desk_annotations = load_desk_annotations(desk_roi_path) # space_id -> {"desk": [x,y,w,h]}

    debug_print(f"Loaded {len(chair_rois_map)} chair ROIs")
    debug_print(f"Loaded {len(desk_annotations)} desk annotations")

    # Create the space mapping
    space_mapping = map_seats_to_spaces(chair_rois_map, desk_annotations)
    debug_print(f"Created {len(space_mapping)} space mappings")
    if not space_mapping:
        print("Error: No spaces defined from chair or desk ROIs. Exiting.")
        return

    # Load baseline image for desk detection
    debug_print(f"Loading baseline image from: {baseline_image_path}")
    baseline_img = cv2.imread(baseline_image_path)
    if baseline_img is None:
        print(f"Error: Could not load baseline image from {baseline_image_path}")
        return

    # Resize baseline image to match video dimensions
    try:
        baseline_img = cv2.resize(baseline_img, (frame_width, frame_height))
        debug_print(f"Resized baseline image to {frame_width}x{frame_height}")
    except cv2.error as e:
        print(f"Error resizing baseline image: {e}")
        return


    # Stores current state for each space: space_id -> {"chair_occupied": bool, "desk_occupied": bool, "chair_confidence": float, "chair_bbox": list}
    space_state = {
        space_id: {"chair_occupied": False, "desk_occupied": False, "chair_confidence": 0.0, "chair_bbox": None}
        for space_id in space_mapping
    }

    # Stores occupancy evidence during person detection cycle: space_id -> {'positive': count, 'total': count}
    occupancy_evidence = {}

    # Store the last known desk detection results to persist between keyframes: space_id -> (occupied, confidence)
    last_desk_results = {space_id: (False, 0.0) for space_id in space_mapping if space_mapping[space_id]["desk_roi"]}

    # Initialize the occupancy timeline
    occupancy_timeline = initialize_occupancy_timeline()
    occupancy_timeline["fps"] = fps  # Set the actual FPS

    # Track the previous occupancy state *for the timeline* to detect changes
    previous_timeline_state = {} # space_id -> {"chair": bool, "desk": bool}


    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True) # Ensure JSON output dir exists
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Initialize detectors
    person_detector = PersonDetector(model=CONFIG["person_model"], confidence=CONFIG["person_confidence"])
    chair_detector = ChairDetector(model=CONFIG["chair_model"], confidence=CONFIG["chair_confidence"]) # Use primary confidence here
    desk_detector = DeskDetector(baseline_img, desk_annotations)

    frame_count = 0
    desk_vis_windows = {}  # Track visualization windows

    # --- Main Loop ---
    while True:
        frame_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize frame for faster processing (only for detections)
        try:
            resized_frame = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
        except cv2.error as e:
             print(f"Frame {frame_count}: Error resizing frame: {e}. Skipping frame.")
             continue

        # --- Chair Detection (runs on chair keyframes) ---
        chair_detections_this_frame = [] # Raw detections from model
        if frame_count == 1 or frame_count % chair_keyframe_interval == 0:
            chair_detections_this_frame = chair_detector.detect_chairs(resized_frame)
            detected_chair_rois_this_keyframe = set()

            # Map detections to space ROIs
            for det in chair_detections_this_frame:
                bbox_resized = det["bbox"]
                center_resized = get_center(bbox_resized)
                # Scale detection center back to original frame coordinates
                center_orig = (center_resized[0] / resize_factor, center_resized[1] / resize_factor)

                for space_id, mapping_data in space_mapping.items():
                    roi_bbox_orig = mapping_data["chair_roi"]
                    if roi_bbox_orig and point_in_bbox(center_orig, roi_bbox_orig):
                        # Store the most recent detection info for this space
                        scaled_bbox_orig = [coord / resize_factor for coord in bbox_resized]
                        space_state[space_id]["chair_confidence"] = det["confidence"]
                        space_state[space_id]["chair_bbox"] = scaled_bbox_orig
                        detected_chair_rois_this_keyframe.add(space_id)
                        # Don't break, a detection might overlap multiple ROIs slightly,
                        # but point_in_bbox should primarily assign it to one. If needed, add logic for best match.
                        # For simplicity now, last ROI containing the center wins.

            # Reset confidence for chairs in ROIs where no chair was detected this keyframe
            for space_id in space_mapping:
                 if space_mapping[space_id]["chair_roi"] and space_id not in detected_chair_rois_this_keyframe:
                     space_state[space_id]["chair_confidence"] = 0.0
                     space_state[space_id]["chair_bbox"] = None # No current bounding box


        # --- Person Detection Cycle ---
        person_detections = []
        person_based_chair_occupancy = {} # Holds person cycle result: space_id -> bool
        cycle_index = (frame_count - 1) % person_cycle_interval # 0-based index within cycle
        is_in_person_cycle_window = 0 <= cycle_index < person_cycle_length

        if is_in_person_cycle_window:
            person_detections = person_detector.detect_people(resized_frame)
            # Scale person detections back to original frame size.
            for person in person_detections:
                person["bbox"] = [coord / resize_factor for coord in person["bbox"]]

            # Initialize/reset evidence accumulation at the start of the cycle window
            if cycle_index == 0:
                occupancy_evidence = {
                    space_id: {'positive': 0, 'total': 0}
                    for space_id in space_mapping if space_mapping[space_id]["chair_roi"]
                }

            # Accumulate evidence for each space with a chair ROI
            for space_id in occupancy_evidence:
                occupancy_evidence[space_id]['total'] += 1
                chair_bbox = space_state[space_id].get("chair_bbox")
                if chair_bbox: # Only check if a chair is currently tracked
                    person_overlap = False
                    for person in person_detections:
                        if calculate_iou(chair_bbox, person["bbox"]) > iou_threshold:
                            occupancy_evidence[space_id]['positive'] += 1
                            person_overlap = True
                            break # Count only one person overlap per chair per frame

            # Determine occupancy based on accumulated evidence at the end of the cycle window
            if cycle_index == person_cycle_length - 1:
                for space_id in occupancy_evidence:
                    if occupancy_evidence[space_id]['total'] > 0:
                        ratio = occupancy_evidence[space_id]['positive'] / occupancy_evidence[space_id]['total']
                        person_based_chair_occupancy[space_id] = ratio >= occupancy_required_ratio
                    else:
                         # If no evidence gathered (e.g., no chair detected), retain previous state
                         person_based_chair_occupancy[space_id] = space_state[space_id]["chair_occupied"]
                # Update the main state *after* the cycle completes
                for space_id, occupied in person_based_chair_occupancy.items():
                     space_state[space_id]["chair_occupied"] = occupied
            else:
                 # During the cycle, just note the person occupancy temporarily if needed for other logic,
                 # but don't update the main state until the end of the cycle.
                 # For now, we calculate the final chair state later using the main state directly.
                 pass

        # --- Desk Detection (runs on desk keyframes) ---
        current_desk_results = {} # Results from this frame's detection run
        if frame_count == 1 or frame_count % desk_keyframe_interval == 0:
            try:
                current_desk_results = desk_detector.detect_all_desks(frame)
                # Update the persistent 'last_desk_results'
                for space_id, result in current_desk_results.items():
                    last_desk_results[space_id] = result

                # Visualize desk detection differences if in debug mode
                if CONFIG.get("debug_mode", False):
                    for space_id_vis in desk_annotations: # Iterate actual annotations for vis
                        if space_mapping.get(space_id_vis, {}).get("desk_roi"): # Check if it's a valid mapped desk space
                            diff_vis = desk_detector.visualize_desk_difference(frame, space_id_vis)
                            if diff_vis is not None:
                                window_name = f"Desk {space_id_vis} Difference"
                                cv2.imshow(window_name, diff_vis)
                                desk_vis_windows[window_name] = True # Track open windows

            except Exception as e:
                print(f"Frame {frame_count}: Error during desk detection: {e}")
                traceback.print_exc()
                # On error, current_desk_results remains empty, logic below uses last_desk_results

        # --- Determine Final Occupancy States for this Frame ---
        current_frame_chair_occupancy = {}
        current_frame_desk_occupancy = {}
        combined_space_occupancy = {}

        for space_id, mapping_data in space_mapping.items():
            # 1. Determine Desk Occupancy
            desk_occupied, desk_confidence = last_desk_results.get(space_id, (False, 0.0))
            current_frame_desk_occupancy[space_id] = desk_occupied
            if desk_occupied and (frame_count == 1 or frame_count % desk_keyframe_interval == 0): # Print only on keyframes where detection ran
                debug_print(f"Frame {frame_count}: Desk {space_id} detected as occupied (Conf: {desk_confidence:.2f})")

            # 2. Determine Chair Occupancy
            final_chair_occupied = False
            if mapping_data["chair_roi"]: # Only if this space has a chair
                # Start with the state determined by the person detection cycle
                person_occupied = space_state[space_id]["chair_occupied"]

                # Check low confidence rule only if person cycle didn't mark as occupied
                low_conf_occupied = False
                chair_conf = space_state[space_id].get("chair_confidence", 0.0)
                # Apply low confidence rule IF person is not detected AND chair confidence is in the low range
                if not person_occupied and chair_low_confidence_threshold > 0 and 0 < chair_conf < chair_low_confidence_threshold:
                     low_conf_occupied = True
                     debug_print(f"Frame {frame_count}: Chair {space_id} marked occupied due to low confidence ({chair_conf:.2f} < {chair_low_confidence_threshold})")

                final_chair_occupied = person_occupied or low_conf_occupied

            current_frame_chair_occupancy[space_id] = final_chair_occupied


            # 3. Determine Combined Space Occupancy
            combined_space_occupancy[space_id] = final_chair_occupied or desk_occupied

            # --- Record Occupancy Changes to Timeline ---
            # Initialize previous state on first frame
            if frame_count == 1:
                previous_timeline_state[space_id] = {
                     "chair": final_chair_occupied,
                     "desk": desk_occupied
                 }
                # Record initial state for both chair and desk
                record_occupancy_change(occupancy_timeline, space_id, "chair", 1, final_chair_occupied)
                record_occupancy_change(occupancy_timeline, space_id, "desk", 1, desk_occupied)
            else:
                # Record change for chair if state differs from previous recorded state
                if space_id in previous_timeline_state and previous_timeline_state[space_id]["chair"] != final_chair_occupied:
                    record_occupancy_change(occupancy_timeline, space_id, "chair", frame_count, final_chair_occupied)
                    previous_timeline_state[space_id]["chair"] = final_chair_occupied

                # Record change for desk if state differs from previous recorded state
                if space_id in previous_timeline_state and previous_timeline_state[space_id]["desk"] != desk_occupied:
                    record_occupancy_change(occupancy_timeline, space_id, "desk", frame_count, desk_occupied)
                    previous_timeline_state[space_id]["desk"] = desk_occupied

        # Update the end frame of all current timeline segments *after* processing all spaces
        update_segment_end_frames(occupancy_timeline, frame_count)

        # --- Visualization ---
        vis_frame = frame.copy() # Draw on the original frame

        # Draw Chair ROIs (dashed blue)
        for space_id, mapping_data in space_mapping.items():
            if mapping_data["chair_roi"]:
                roi_bbox = mapping_data["chair_roi"]
                pt1 = (int(roi_bbox[0]), int(roi_bbox[1]))
                pt2 = (int(roi_bbox[2]), int(roi_bbox[3]))
                draw_dashed_rectangle(vis_frame, pt1, pt2, (255, 0, 0), thickness=1, dash_length=8)
                # cv2.putText(vis_frame, f"C_ROI:{space_id}", (pt1[0], pt1[1] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Draw Desk ROIs (dashed, color based on occupancy)
        for space_id, mapping_data in space_mapping.items():
            if mapping_data["desk_roi"]:
                x, y, w, h = mapping_data["desk_roi"]
                pt1 = (int(x), int(y))
                pt2 = (int(x + w), int(y + h))
                is_occupied = current_frame_desk_occupancy.get(space_id, False)
                color = (0, 0, 255) if is_occupied else (0, 255, 0) # Red if occupied, green if empty
                draw_dashed_rectangle(vis_frame, pt1, pt2, color, thickness=1, dash_length=8)
                status = "Occ" if is_occupied else "Emp"
                # cv2.putText(vis_frame, f"D_ROI:{space_id}({status})", (pt1[0], pt1[1] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw Tracked/Detected Chairs (solid rectangle, color based on occupancy)
        for space_id, state in space_state.items():
            if state["chair_bbox"]: # Only draw if a chair is currently detected/tracked in this space
                bbox = state["chair_bbox"]
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                is_occupied = current_frame_chair_occupancy.get(space_id, False)
                color = (0, 0, 255) if is_occupied else (0, 255, 0) # Red if occupied, green if empty
                cv2.rectangle(vis_frame, pt1, pt2, color, 2)
                status = "Occupied" if is_occupied else "Empty"
                conf_str = f" ({state['chair_confidence']:.2f})" if state['chair_confidence'] > 0 else ""
                cv2.putText(vis_frame, f"Chair {space_id}: {status}{conf_str}", (pt1[0], pt1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Person Detections (yellow, only if in detection window)
        if is_in_person_cycle_window:
            for person in person_detections:
                bbox = person["bbox"]
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(vis_frame, pt1, pt2, (0, 255, 255), 1)
                # cv2.putText(vis_frame, f"P {person['confidence']:.2f}", (pt1[0], pt1[1] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw combined space occupancy status at the top right
        y_pos = 30
        sorted_space_ids = sorted(combined_space_occupancy.keys()) # Sort for consistent display order
        for space_id in sorted_space_ids:
            occupied = combined_space_occupancy[space_id]
            status = "OCCUPIED" if occupied else "EMPTY"
            color = (0, 0, 255) if occupied else (0, 255, 0)
            text = f"{space_id}: {status}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            x_pos = frame_width - text_size[0] - 10
            cv2.putText(vis_frame, text, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25 # Adjust spacing

        # Calculate and show processing time
        frame_process_time = (time.time() - frame_start_time) * 1000  # ms
        cv2.putText(vis_frame, f"Frame: {frame_count} | Proc Time: {frame_process_time:.1f} ms",
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Combined Occupancy Monitoring", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video
        video_writer.write(vis_frame)

    # --- End of Loop ---

    # Finalize the timeline
    finalize_timeline(occupancy_timeline, frame_count)

    # Write the timeline to a JSON file
    try:
        with open(json_output_path, 'w') as f:
            json.dump(occupancy_timeline, f, indent=2)
        print(f"Occupancy timeline successfully written to {json_output_path}")
    except Exception as e:
        print(f"Error writing JSON output to {json_output_path}: {e}")

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    # Close specific desk difference windows if open
    for window_name in desk_vis_windows:
        cv2.destroyWindow(window_name)

    print("Processing finished.")


if __name__ == "__main__":
    process_video()
