#!/usr/bin/env python3
"""
Simplified Occupancy Monitoring with Chair Tracking (Real-Time Visualization)
"""

import os
import cv2
import json
import time
import logging
from collections import deque
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Configuration
class Config:
    video_path = 'data/videos/video_1.MOV'
    output_dir = 'output_2'
    resize_factor = 0.5
    keyframe_interval = 30
    chair_model = 'yolov8m_chair_cpu'
    person_model = 'yolov8m'
    confidence = 0.4
    iou_threshold = 0.15
    max_chair_age = 120
    temporal_window = 5
    display = True  # Added display flag
    colors = {
        'empty': (0, 255, 0),
        'occupied': (0, 165, 255),
        'person': (0, 255, 255)
    }

class Detector:
    def __init__(self, model_type, model_path, conf_thresh):
        self.model = YOLO(self.get_model_path(model_type, model_path))
        self.conf_thresh = conf_thresh

    @staticmethod
    def get_model_path(model_type, model_path):
        if model_path and model_path.endswith('.pt'):
            return model_path
        return f'models/{model_type}.pt'

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        return [
            {'bbox': box.xyxy[0].tolist(), 'conf': float(box.conf[0])}
            for box in (results.boxes if results else [])
            if float(box.conf[0]) >= self.conf_thresh
        ]

class ChairTracker:
    def __init__(self):
        self.chairs = {}
        self.next_id = 0

    def update(self, detections, frame_size):
        current = {}
        for det in detections:
            chair_id = self._match_existing(det['bbox'])
            if chair_id is None:
                chair_id = self.next_id
                self.next_id += 1

            current[chair_id] = {
                'bbox': self.scale_bbox(det['bbox'], frame_size),
                'history': deque([False]*Config.temporal_window, maxlen=Config.temporal_window),
                'age': 0
            }
        
        self.chairs = {cid: self._update_chair(cid, chair) for cid, chair in current.items()}

    def _update_chair(self, cid, chair):
        chair['age'] += 1
        return {**self.chairs.get(cid, {}), **chair}

    def _match_existing(self, bbox):
        for cid, chair in self.chairs.items():
            if self.iou(chair['bbox'], bbox) > 0.5:
                return cid
        return None

    @staticmethod
    def scale_bbox(bbox, scale_factor):
        return [int(coord * scale_factor) for coord in bbox]

    @staticmethod
    def iou(box1, box2):
        x1, y1, x2, y2 = (max(box1[i], box2[i]) for i in range(4))
        inter = max(0, x2-x1) * max(0, y2-y1)
        area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1])
        return inter / (area - inter) if area else 0

def process_video():
    # Initialize components
    cap = cv2.VideoCapture(Config.video_path)
    chair_detector = Detector(Config.chair_model, None, Config.confidence)
    person_detector = Detector(Config.person_model, None, Config.confidence)
    tracker = ChairTracker()
    
    # Setup output
    os.makedirs(Config.output_dir, exist_ok=True)
    writer = cv2.VideoWriter(f'{Config.output_dir}/output.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Detection
        resized = cv2.resize(frame, None, fx=Config.resize_factor, fy=Config.resize_factor)
        chairs = chair_detector.detect(resized) if frame_count % Config.keyframe_interval == 0 else []
        persons = person_detector.detect(resized)
        
        # Tracking
        tracker.update(chairs, 1/Config.resize_factor)
        
        # Occupancy check
        for cid, chair in tracker.chairs.items():
            occupied = any(tracker.iou(chair['bbox'], p['bbox']) > Config.iou_threshold for p in persons)
            chair['history'].append(occupied)
            chair['occupied'] = sum(chair['history']) / len(chair['history']) > 0.5
        
        # Visualization
        display_frame = frame.copy()
        
        # Draw persons
        for p in persons:
            x1, y1, x2, y2 = [int(coord * (1/Config.resize_factor)) for coord in p['bbox']]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), Config.colors['person'], 2)
        
        # Draw chairs
        for cid, chair in tracker.chairs.items():
            color = Config.colors['occupied'] if chair['occupied'] else Config.colors['empty']
            x1, y1, x2, y2 = map(int, chair['bbox'])
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, f"{cid}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add FPS counter
        fps = 1 / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        if Config.display:
            cv2.imshow('Occupancy Monitoring', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Write output
        writer.write(display_frame)
        
        # Performance logging
        if frame_count % 30 == 0:
            logging.info(f"Frame {frame_count} processed in {1/fps:.2f}s")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_video()