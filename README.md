# Campus Occupancy Detection System

## Project Overview

This repository contains a computer vision-based system for monitoring seat occupancy in university libraries and study spaces, developed as part of the MDM3 module at the University of Bristol Engineering Mathematics Course. The system uses a combination of background subtraction techniques and deep learning models to accurately detect and track seat occupancy in real-time.

## Introduction to Computer Vision for Occupancy Monitoring

Computer vision offers powerful tools for understanding and analyzing space utilization without requiring physical sensors at each seat. Our approach combines traditional computer vision techniques with modern deep learning to create a robust, privacy-conscious monitoring system that provides valuable insights into how campus spaces are utilized.

### Key Features

- **Real-time occupancy detection** using multiple computer vision techniques
- **Privacy-preserving** implementation that doesn't store identifiable images of individuals
- **ROI (Region of Interest)** system for monitoring specific seats and desk areas
- **Multi-method validation** to reduce false positives and increase accuracy
- **Visual output** showing occupancy status across monitored areas

## Tracking and Counting Methodologies

The system employs two primary approaches to detect seat occupancy:

1. **Background Subtraction Analysis**: Detects changes within defined ROIs by comparing current frames with an established baseline (empty) image.
2. **Person Detection Validation**: Uses pre-trained YOLO models to validate occupancy by detecting people in chairs.

These methodologies work in tandem to provide high-accuracy detection while minimizing false positives from environmental changes.

## Tracking Algorithms

Several specialized tracking and detection algorithms have been implemented:

- **MOG2 Background Subtraction**: Adapts to scene changes while detecting foreground objects.
- **HSV-based Detection**: Analyzes color space changes to detect occupancy.
- **Histogram-based Comparison**: Compares color distribution changes between baseline and current frames.
- **Radial Mask with Gaussian Weighting**: Applies weighted importance to central areas of each ROI.
- **Combined Approach**: Integrates multiple methods with a voting system for maximum accuracy.

## Implementation Challenges and Solutions

### Lighting Variations

**Challenge**: Library lighting changes throughout the day, affecting detection accuracy.

**Solution**: Multiple baseline images for different lighting conditions and HSV color space analysis to reduce sensitivity to illumination changes.

### Occlusion Issues

**Challenge**: People partially hidden behind monitors or other furniture.

**Solution**: Dual-ROI system (chair and desk) to detect occupancy even when one area is partially occluded.

### False Positives

**Challenge**: Movement of chairs or temporary objects triggering false detections.

**Solution**: Multi-criteria validation requiring both background changes and person detection before confirming occupancy.

## Floor Plan Integration

The system integrates with floor plans to provide spatial context for occupancy data

## Desk Monitoring System

### Initial Approach: Full Background Subtraction

Our initial implementation used full-frame background subtraction, which faced several challenges:

- High sensitivity to minor movements (papers shuffling, slight chair movements).
- Difficulty adapting to gradual lighting changes.
- Computational intensity for high-resolution video.

### Pre-trained Model Integration

To improve accuracy, we incorporated pre-trained YOLOv8 models:

- **YOLOv8m** model fine-tuned on chair occupancy data.
- Provides secondary validation of occupancy detected by background subtraction.
- Reduces false positives by requiring person detection within chair ROIs.

We also experimented with fine-tuning this model to work on specific library layouts.

### Mixed Model with Gaussian Masking

1. ROI-specific background subtraction with adaptive parameters.
2. Gaussian weighted importance masks that prioritize central areas of ROIs.
3. Pre-trained YOLO model validation for person detection.
4. Temporal smoothing to prevent rapid oscillation between states.

## Getting Started

### Prerequisites

- Python 3.8+
- OpenCV 4.5+
- NumPy
- PyTorch (for YOLO models)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/campus-occupancy.git
cd campus-occupancy

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python download_models.py
```

### Usage

1. **Prepare your environment**:
   ```bash
   python src/visualize_rois.py  # Visualize and verify ROIs
   ```

2. **Run background subtraction analysis**:
   ```bash
   python src/background_subtraction_combined.py
   ```

3. **Run the complete occupancy monitor**:
   ```bash
   python src/occupancy_monitor_tier1_improved.py
   ```

## Project Structure

- `/annotations`: Contains JSON files defining ROIs for chairs and desks.
- `/data`: Test images and videos for system evaluation.
- `/models`: Pre-trained YOLO and custom models.
- `/src`: Source code for all detection algorithms.
- `/output`: Results and processed videos.

## Documentation

- [Manual ROI Labelling Guide](Manual_ROI_labelling.md)
- [Data Collection Guide](Data_Collection.md)
- [Project Overview](Project_Overview.md)

## Contributors

- University of Bristol MDM3 Team

## License

No License
