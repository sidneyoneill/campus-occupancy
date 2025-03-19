import os
import cv2
import json
import time
import numpy as np

from collections import defaultdict

# Instead of importing PersonDetector, we define an integrated person tracker below.
from ultralytics import YOLO

# -----------------------------
# Configuration Settings
# -----------------------------
CONFIG = {
    # Input/Output Paths
    "video_path": "data/videos/video_1.MOV",
    "output_path": "output/processed_video_4.MOV",
    "roi_path": "annotations/chair_locations_2.json",
    
    # Processing Parameters
    "keyframe_interval": 30,    # Do new chair detection every N frames
    "iou_threshold": 0.15,      # IoU threshold for person overlap with chair
    "resize_factor": 0.5,       # To speed up processing
    
    # Person Detection/Tracking
    "person_model": "yolov8m",  # Model for person detection
    "person_confidence": 0.15,  # Confidence threshold for person detection
    "use_tracking": False,      # Whether to use tracking functionality
    "tracker": "bytetrack.yaml", # Tracker configuration
    
    # Chair Detection
    "chair_model": "yolov8m_chair_cpu",  # Model for chair detection
    "chair_confidence": 0.4,    # Confidence threshold for chair detection
}

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

class PersonTracker:
    """
    A YOLO-based person tracker using integrated tracking functionality.
    This class leverages YOLOv8's built-in tracking when calling predict().
    """
    def __init__(self, model=CONFIG["person_model"], confidence=CONFIG["person_confidence"], 
                 tracker=CONFIG["tracker"], use_tracking=CONFIG["use_tracking"]):
        if model.endswith('.pt'):
            model_path = model
        else:
            model_path = os.path.join("models", f"{model}.pt")
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print(f"Model not found at {model_path}. Please ensure the model is available.")
        self.model = YOLO(model_path)
        self.confidence = confidence
        # You can specify a tracker configuration file if needed.
        # By default, YOLOv8 uses a built-in tracker if tracking=True is passed.
        self.tracker = tracker
        self.use_tracking = use_tracking

    def track_people(self, frame):
        """
        Run YOLO prediction with tracking enabled if use_tracking is True.
        Returns a list of person detections with keys "bbox", "confidence", and "id".
        """
        if self.use_tracking:
            # The integrated tracking functionality is enabled by passing tracking=True.
            results = self.model.predict(source=frame, conf=self.confidence, tracker=self.tracker, verbose=False)
        else:
            # Standard detection without tracking
            results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
            
        people = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == 0:  # 0 is the class ID for 'person' in COCO dataset
                    bbox = box.xyxy[0].tolist()
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    # When not using tracking, assign a default ID of -1
                    track_id = int(box.id[0]) if self.use_tracking and hasattr(box, "id") and box.id is not None else -1
                    people.append({"bbox": bbox, "confidence": conf, "id": track_id})
        return people

# -----------------------------
# Main Processing Function
# -----------------------------

def process_video(use_tracking=CONFIG["use_tracking"]):
    # Configuration settings
    video_path = CONFIG["video_path"]
    output_path = CONFIG["output_path"]
    roi_path = CONFIG["roi_path"]
    keyframe_interval = CONFIG["keyframe_interval"]
    iou_threshold = CONFIG["iou_threshold"]
    resize_factor = CONFIG["resize_factor"]

    # Load chair ROIs from file
    chair_rois = load_chair_rois(roi_path)
    # The chair tracker maps an ROI id to the latest detection:
    # { "bbox": [...], "confidence": ..., "occupied": bool }
    chair_tracker = {}

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

    # Initialize detectors
    # Pass the tracking flag to the person tracker
    person_tracker = PersonTracker(model=CONFIG["person_model"], 
                                  confidence=CONFIG["person_confidence"], 
                                  use_tracking=use_tracking)
    chair_detector = ChairDetector(model=CONFIG["chair_model"], 
                                  confidence=CONFIG["chair_confidence"])

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (int(frame_width * resize_factor), int(frame_height * resize_factor)))

        # On the first frame and every keyframe interval, run chair detection.
        if frame_count == 1 or frame_count % keyframe_interval == 0:
            detections = chair_detector.detect_chairs(resized_frame)
            # For each detection, update the chair tracker if its center lies within any ROI.
            for det in detections:
                bbox = det["bbox"]
                center = get_center(bbox)
                # Scale detection center back to original frame coordinates.
                center = (center[0] / resize_factor, center[1] / resize_factor)
                for roi_id, roi_bbox in chair_rois.items():
                    if point_in_bbox(center, roi_bbox):
                        # Always keep the most recent detection for this ROI.
                        scaled_bbox = [coord / resize_factor for coord in bbox]
                        chair_tracker[roi_id] = {"bbox": scaled_bbox,
                                                 "confidence": det["confidence"],
                                                 "occupied": False}
                        # Once updated for a matching ROI, break out of the loop.
                        break

        # Run person tracking on every frame using integrated tracking.
        person_detections = person_tracker.track_people(resized_frame)
        # Scale person detections back to original frame size.
        for person in person_detections:
            person["bbox"] = [coord / resize_factor for coord in person["bbox"]]

        # Update occupancy: for each tracked chair (by ROI), check overlap with person bounding boxes.
        for roi_id, chair in chair_tracker.items():
            chair_bbox = chair["bbox"]
            occupied = False
            for person in person_detections:
                if calculate_iou(chair_bbox, person["bbox"]) > iou_threshold:
                    occupied = True
                    break
            chair["occupied"] = occupied

        # Visualization on the original frame.
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

        # Optionally, draw person detections (displayed here in yellow along with their track ID).
        for person in person_detections:
            bbox = person["bbox"]
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(vis_frame, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"ID {person['id']} {person['confidence']:.2f}", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Optionally, show frame count.
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add a text indicator for the tracking mode
        tracking_mode = "Tracking ON" if use_tracking else "Tracking OFF"
        cv2.putText(vis_frame, tracking_mode, (frame_width - 200, frame_height - 10),
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
    # Use the tracking setting from the CONFIG
    process_video(CONFIG["use_tracking"])
