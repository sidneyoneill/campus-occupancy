# Data Collection and Preparation Guide for Seat Occupancy Detection

## Overview

This document details the methodology used for collecting and preparing the dataset for our seat occupancy detection system. High-quality data collection was crucial for developing an accurate and robust detection model that can handle various real-world scenarios.

## Equipment Setup

### Camera Equipment
- **Device**: Smartphone camera (iPhone) with high-resolution video and photo capabilities
- **Mounting Options**:
  - Handheld recording for flexibility in capturing different angles
  - Tripod-mounted for stability during longer recordings
  - DIY ceiling mount using adhesive hooks and smartphone holder for consistent overhead perspective

### Room Configuration
- Standard meeting room or office space with a table and multiple chairs
- Camera positioned to capture a bird's-eye view of the entire seating area
- Sufficient height (approximately 2-2.5 meters) to ensure all seats were visible in frame

## Data Collection Scenarios

We systematically captured the following scenarios to ensure comprehensive training data:

### 1. Empty State Baseline
- All seats completely empty
- Multiple recordings under different lighting conditions
- Static images from various angles to establish baseline appearance

### 2. Partial Occupancy Scenarios
- One person occupying a single seat
- Multiple people occupying different combinations of seats
- People sitting in different postures (upright, leaning, slouching)
- Recordings with people working at the table (typing, writing, reading)

### 3. Transition Events
- People entering the frame and sitting down
- People standing up and leaving seats
- People temporarily leaving and returning to the same seat
- Chair movement and repositioning

### 4. Lighting Variations
- Natural daylight (morning, midday, afternoon)
- Artificial lighting only
- Mixed lighting conditions
- Shadows and lighting transitions (e.g., clouds passing, lights being turned on/off)

## Recording Specifications

### Video Recordings
- **Duration**: 1-2 minutes per scenario
- **Resolution**: 1080p or higher
- **Frame Rate**: 30fps
- **Format**: MP4/MOV
- **Focus**: Ensured camera remained focused on the seating area throughout recording
- **Stability**: Minimized camera shake, especially for transition events

### Static Images
- **Resolution**: 12MP or higher
- **Format**: JPEG/PNG
- **Quantity**: 10-15 images per scenario
- **Angles**: Primarily overhead, with some supplementary angled views
- **Lighting**: Consistent exposure settings within each lighting scenario

## Data Organization

The collected data was organized using the following structure:

1. **Videos Folder**:
   - Subfolders for each scenario (empty, partial, transitions)
   - Further categorized by lighting conditions
   - Naming convention: `[scenario]_[lighting]_[date]_[sequence].mp4`

2. **Images Folder**:
   - Categorized by occupancy state and lighting
   - Naming convention: `[occupancy]_[lighting]_[date]_[sequence].jpg`

3. **Metadata File**:
   - CSV document tracking all recordings with details on:
     - Date and time of recording
     - Lighting conditions
     - Number of people present
     - Specific seats occupied
     - Any notable events or anomalies

## Data Preprocessing Steps

Before using the collected data for model training:

1. **Video Frame Extraction**:
   - Extracted key frames at 1-second intervals
   - Additional frames extracted during transition events (0.2-second intervals)

2. **Image Standardization**:
   - Cropped images to focus on the seating area
   - Resized to uniform dimensions (1280×720 pixels)
   - Adjusted brightness and contrast for consistency across lighting conditions

3. **Manual Annotation**:
   - Defined ROIs for each chair and corresponding desk area
   - Labeled occupancy state for each seat in every frame
   - Created ground truth bounding boxes around seated individuals

## Quality Assurance

To ensure dataset quality, we implemented the following checks:

- Visual inspection of all recordings for blur, obstruction, or poor framing
- Verification that all required scenarios were adequately represented
- Confirmation that lighting variations were sufficiently diverse
- Validation that transition events were clearly captured
- Review of annotations for accuracy and consistency

## Challenges and Solutions

- **Lighting Inconsistency**: Used post-processing to normalize extreme variations
- **Camera Stability**: Implemented additional stabilization for handheld recordings
- **Occlusion Issues**: Ensured multiple angles were captured for ambiguous situations
- **Privacy Concerns**: Obtained consent from all individuals appearing in the dataset

This comprehensive data collection approach provided a robust foundation for developing our seat occupancy detection system, ensuring it can perform reliably across various real-world conditions.
