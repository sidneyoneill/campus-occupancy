# Seat Occupancy Detection System: Project Objectives

## Project Overview

This document outlines the objectives and specifications for developing a seat occupancy detection system using manual Region of Interest (ROI) labeling. The system aims to accurately detect whether seats are occupied in real-time using computer vision techniques.

## Detection Methodology

The system will employ a dual-criteria approach to determine seat occupancy:

### 1. Background Subtraction Analysis

- Each seat will have two manually defined ROIs:
  - **Chair ROI**: Covering the seat surface and backrest
  - **Desk ROI**: Encompassing the area immediately in front of the chair

- The system will perform background subtraction within these ROIs to detect changes that indicate potential occupancy.
- This approach is sensitive to movement and presence within the defined regions.

### 2. Person Detection Validation

- A pre-trained YOLO (You Only Look Once) model will be employed for person detection.
- Occupancy will be confirmed when:
  - A person is detected by the YOLO model
  - The person's bounding box significantly overlaps with the chair ROI
  
- This secondary validation helps reduce false positives from the background subtraction method.

## Performance Requirements

- **Real-time Processing**: The system must operate in real-time on standard consumer hardware (e.g., MacBook M1).
- **Frame Rate**: Minimum 15 FPS processing capability.
- **Latency**: Detection delay should not exceed 200ms.

## Technical Considerations

### Optimization Strategies

- Balance between speed and accuracy:
  - Optimize background subtraction parameters for efficiency
  - Consider using a lightweight YOLO variant (like YOLOv5-small or YOLOv8-nano)
  - Implement multi-threading for parallel processing of background subtraction and YOLO detection
  - Use ROI-specific processing to reduce computational load

### Accuracy Goals

- **False Positive Rate**: < 5%
- **False Negative Rate**: < 2%
- **Overall Accuracy**: > 95% under normal lighting conditions

## Implementation Phases

1. ROI Definition Interface
2. Background Subtraction Implementation
3. YOLO Integration
4. Detection Logic Fusion
5. Performance Optimization
6. Testing and Validation

## Future Considerations

- Adaptive background modeling for changing lighting conditions
- Extension to multi-camera setups
- Privacy-preserving implementations
