import os
import cv2
import json
import time
import numpy as np

from collections import defaultdict

# -----------------------------
# Configuration Settings
# -----------------------------
CONFIG = {
    # Input/Output Paths
    "video_path": "data/videos/video_1.MOV",
    "output_path": "output/processed_video_7.MOV",
    "roi_path": "annotations/chair_locations_2.json",
    
    # Processing Parameters
    "keyframe_interval": 100,       # Do new chair detection every N frames
    "iou_threshold": 0.15,         # IoU threshold for person overlap with chair
    "resize_factor": 0.5,          # To speed up processing

    # Person Detection Cycle Parameters
    "person_cycle_interval": 100,  # Every 100 frames start a new detection cycle
    "person_cycle_length": 10,     # Run person detection for 10 consecutive frames in the cycle
    "occupancy_required_ratio": 0.3,  # If occupancy evidence in >= 30% of detection frames, mark chair occupied

    # Person Detection
    "person_model": "yolov8m",     # Model for person detection
    "person_confidence": 0.1,      # Confidence threshold for person detection
    
    # Chair Detection
    "chair_model": "yolov8m_chair_cpu",  # Model for chair detection
    "chair_confidence": 0.4,       # Confidence threshold for chair detection
}

# Import the person detection and chair detection modules.
# (Ensure these modules are available in your project.)
from person_detector import PersonDetector
from ultralytics import YOLO

# -----------------------------
# Helper Classes and Functions
# -----------------------------

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
                converted[key] = [x, y, x + w, y + h]
        return converted
    except Exception as e:
        print("Error loading ROIs:", e)
        return {}

# -----------------------------
# Main Processing Function
# -----------------------------

def process_video():
    # Configuration settings from CONFIG dictionary
    video_path = CONFIG["video_path"]
    output_path = CONFIG["output_path"]
    roi_path = CONFIG["roi_path"]
    keyframe_interval = CONFIG["keyframe_interval"]
    iou_threshold = CONFIG["iou_threshold"]
    resize_factor = CONFIG["resize_factor"]

    person_cycle_interval = CONFIG["person_cycle_interval"]
    person_cycle_length = CONFIG["person_cycle_length"]
    occupancy_required_ratio = CONFIG["occupancy_required_ratio"]

    # Load chair ROIs from file
    chair_rois = load_chair_rois(roi_path)
    # The chair tracker maps an ROI id to the latest detection:
    # { "bbox": [...], "confidence": ..., "occupied": bool }
    chair_tracker = {}

    # Initialize occupancy evidence dictionary for person detections.
    # Structure: { roi_id: {'positive': count, 'total': count} }
    occupancy_evidence = {}

    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Initialize detectors using CONFIG parameters
    person_detector = PersonDetector(model=CONFIG["person_model"], confidence=CONFIG["person_confidence"])
    chair_detector = ChairDetector(model=CONFIG["chair_model"], confidence=CONFIG["chair_confidence"])

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (int(frame_width * resize_factor), int(frame_height * resize_factor)))

        # --- Chair Detection (runs on keyframes) ---
        if frame_count == 1 or frame_count % keyframe_interval == 0:
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
        else:
            # Not in active detection window.
            person_detections = []

        # At the end of the active detection window, update chair occupancy based on evidence.
        if cycle_index == person_cycle_length - 1:
            for roi_id, evidence in occupancy_evidence.items():
                ratio = evidence['positive'] / evidence['total'] if evidence['total'] > 0 else 0
                if ratio >= occupancy_required_ratio:
                    chair_tracker[roi_id]["occupied"] = True
                else:
                    chair_tracker[roi_id]["occupied"] = False
            # Clear evidence for next cycle.
            occupancy_evidence = {}

        # --- Visualization ---
        vis_frame = frame.copy()
        # Draw each ROI as a dashed blue rectangle.
        for roi_id, roi_bbox in chair_rois.items():
            pt1 = (int(roi_bbox[0]), int(roi_bbox[1]))
            pt2 = (int(roi_bbox[2]), int(roi_bbox[3]))
            draw_dashed_rectangle(vis_frame, pt1, pt2, (255, 0, 0), thickness=2, dash_length=10)
            cv2.putText(vis_frame, f"ROI: {roi_id}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw tracked chairs as solid rectangles along with occupancy status.
        for roi_id, chair in chair_tracker.items():
            bbox = chair["bbox"]
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            # Change color based on occupancy: red if occupied, green if empty.
            color = (0, 0, 255) if chair["occupied"] else (0, 255, 0)
            cv2.rectangle(vis_frame, pt1, pt2, color, 2)
            status = "Occupied" if chair["occupied"] else "Empty"
            cv2.putText(vis_frame, f"Chair [{roi_id}]: {status}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Optionally, draw person detections (if any) in yellow.
        for person in person_detections:
            bbox = person["bbox"]
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(vis_frame, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Person {person['confidence']:.2f}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show frame count (optional)
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Simplified Occupancy Monitoring", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write the frame to the output video.
        video_writer.write(vis_frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
