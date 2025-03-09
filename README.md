# Seat Occupancy Detection System

## Overview
This project implements a seat occupancy detection system using computer vision techniques. The system detects whether seats are occupied based on two criteria:
1. Background subtraction in manually defined ROIs
2. Person detection using a pre-trained YOLO model

## Project Structure
- `data/`: Contains input data
  - `images/`: Static photos of the seating area
  - `videos/`: Recorded videos for testing and training
- `annotations/`: ROI definitions for seats (JSON/YAML)
- `models/`: Pre-trained and fine-tuned model files
- `src/`: Python source code
  - `person_detector.py`: Module for detecting people using YOLO

## Person Detection
The system uses Ultralytics YOLO (YOLOv8) for person detection:

- **Model**: YOLOv8s (small variant) pre-trained on COCO dataset
- **Features**:
  - Detects people in images with configurable confidence threshold
  - Filters detections to only include the 'person' class
  - Returns bounding boxes and confidence scores for each detected person
  - Includes visualization tools for debugging and analysis

### Using the Person Detector

```python
from src.person_detector import PersonDetector
import cv2

# Initialize the detector
detector = PersonDetector(confidence=0.5)  # Adjust confidence threshold as needed

# Load an image
image = cv2.imread('path/to/image.jpg')

# Detect people
detections = detector.detect_people(image)

# Count people
count = detector.count_people(image)
print(f"Detected {count} people in the image")

# Visualize detections
output_image = detector.visualize_detections(image, detections)
cv2.imwrite('output.jpg', output_image)
```

### Command Line Usage

```bash
python src/person_detector.py --image path/to/image.jpg --output output.jpg --conf 0.5
```

## Setup
1. Create a virtual environment:
   ```
   python -m venv seat_detection_env
   ```

2. Activate the virtual environment:
   ```
   source seat_detection_env/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the detection system:
   ```
   python src/main.py
   ```

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)

See `requirements.txt` for specific version requirements.

## Integration with Occupancy Pipeline

The person detector can be integrated into the broader occupancy detection system:

1. **Real-time monitoring**: Process camera feeds to count people in different areas
2. **Occupancy analytics**: Track occupancy patterns over time
3. **Alert system**: Generate notifications when occupancy exceeds thresholds
4. **Visualization**: Create heatmaps and dashboards for occupancy data

Example integration:

```python
import cv2
from src.person_detector import PersonDetector

# Initialize detector
detector = PersonDetector()

# Connect to camera feed
cap = cv2.VideoCapture(0)  # Use camera index or RTSP URL

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Count people in frame
    count = detector.count_people(frame)
    
    # Process occupancy data
    # ...
    
    # Display result
    cv2.putText(frame, f"Occupancy: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Occupancy Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
