# Manual ROI Annotation Guide for Seat Occupancy Detection System

## Overview

This document provides a comprehensive guide for manually annotating Regions of Interest (ROIs) for our seat occupancy detection system. Each seat requires two distinct ROIs: one for the chair itself and one for the corresponding desk area. These annotations serve as the foundation for our dual-criteria detection approach.

## Prerequisites

- Collected overhead images of the seating area (as described in the data collection guide)
- One of the following annotation tools:
  - LabelImg (recommended for rectangular ROIs)
  - CVAT (for more complex polygon shapes)
  - Our custom OpenCV-based annotator tool
- Python environment with OpenCV, NumPy, and JSON/PyYAML libraries installed
- Reference image showing all seats clearly visible and unoccupied

## Annotation Process

### Step 1: Prepare Your Environment

1. Select a clear, high-resolution reference image from your dataset that shows all seats unoccupied.
2. Launch your chosen annotation tool:
   - For LabelImg: `labelimg [path-to-reference-image]`
   - For custom OpenCV annotator: `python annotator.py --image [path-to-reference-image]`

### Step 2: Create a Consistent Naming Convention

Establish a naming convention for seat IDs and ROIs:
- Seat IDs: `seat_1`, `seat_2`, etc. (numbered clockwise from the top-left)
- Chair ROIs: `seat_[number]_chair`
- Desk ROIs: `seat_[number]_desk`

### Step 3: Annotate Chair ROIs

For each seat in the reference image:

1. Select the polygon or rectangle tool in your annotation software.
2. For the chair ROI:
   - Draw a region covering the entire seat surface where a person's lower body would be positioned.
   - Include the seat pan and backrest if visible from overhead.
   - Ensure the ROI is tight to the chair boundaries but includes the full seating area.
   - Label this ROI as `seat_[number]_chair` (e.g., `seat_1_chair`).

3. Verify the chair ROI by checking that:
   - The entire seating surface is covered
   - The ROI doesn't extend significantly beyond the chair's physical boundaries
   - The annotation is precise enough to distinguish adjacent chairs

### Step 4: Annotate Desk ROIs

For each seat's corresponding desk area:

1. Select the polygon or rectangle tool again.
2. For the desk ROI:
   - Draw a region covering the desk area directly in front of the chair.
   - This should encompass where a seated person's upper body, arms, and any items they might use (laptop, papers) would typically be positioned.
   - Extend the region approximately 40-60cm from the edge of the chair toward the table center.
   - Label this ROI as `seat_[number]_desk` (e.g., `seat_1_desk`).

3. Verify the desk ROI by checking that:
   - It covers the working area in front of the chair
   - It doesn't significantly overlap with adjacent seats' desk areas
   - The size is proportional to typical upper body movements when seated

### Step 5: Review and Refine Annotations

1. After completing all annotations, review each ROI for accuracy:
   - Ensure all seats have both chair and desk ROIs defined
   - Check for any gaps or excessive overlaps between adjacent ROIs
   - Verify that ROIs are appropriately sized for their purpose

2. Make adjustments as needed:
   - Resize or reshape ROIs that are too large or small
   - Reposition ROIs that don't align properly with the furniture
   - Ensure consistent sizing across similar chairs/desk areas

### Step 6: Export Annotations

#### Using LabelImg:

1. Save annotations in XML format:
   - File → Save
   - This creates an XML file with all ROI coordinates

2. Convert the XML to our required JSON/YAML format using the provided conversion script:
   ```
   python convert_annotations.py --input annotations.xml --output seat_config.json
   ```

#### Using Custom OpenCV Annotator:

1. When finished annotating, press 'S' to save directly to JSON/YAML format.
2. Verify the output file contains all annotations correctly formatted.

### Step 7: Format the Configuration File

Ensure your configuration file follows this structure:

```json
{
  "seats": {
    "seat_1": {
      "chair_roi": {
        "type": "polygon",
        "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
      },
      "desk_roi": {
        "type": "polygon",
        "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
      }
    },
    "seat_2": {
      "chair_roi": {
        "type": "polygon",
        "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
      },
      "desk_roi": {
        "type": "polygon",
        "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
      }
    },
    // Additional seats...
  },
  "metadata": {
    "image_width": 1920,
    "image_height": 1080,
    "annotation_date": "2023-10-15",
    "annotator": "Your Name"
  }
}
```

For YAML format:

```yaml
seats:
  seat_1:
    chair_roi:
      type: polygon
      points: [[x1, y1], [x2, y2], ..., [xn, yn]]
    desk_roi:
      type: polygon
      points: [[x1, y1], [x2, y2], ..., [xn, yn]]
  seat_2:
    chair_roi:
      type: polygon
      points: [[x1, y1], [x2, y2], ..., [xn, yn]]
    desk_roi:
      type: polygon
      points: [[x1, y1], [x2, y2], ..., [xn, yn]]
  # Additional seats...
metadata:
  image_width: 1920
  image_height: 1080
  annotation_date: "2023-10-15"
  annotator: "Your Name"
```

### Step 8: Validate the Configuration File

1. Run the validation script to ensure the configuration file is properly formatted:
   ```
   python validate_roi_config.py --config seat_config.json
   ```

2. The script will check for:
   - Presence of both ROIs for each seat
   - Valid coordinate values within image dimensions
   - Proper formatting of all entries
   - Reasonable ROI sizes and positions

3. Fix any issues reported by the validation script.

### Step 9: Visualize the Annotations

1. Run the visualization tool to confirm your annotations look correct:
   ```
   python visualize_rois.py --config seat_config.json --image [path-to-reference-image]
   ```

2. The tool will display:
   - The reference image with all ROIs overlaid
   - Chair ROIs in one color (e.g., blue)
   - Desk ROIs in another color (e.g., green)
   - Seat IDs labeled next to each ROI pair

3. Save this visualization for documentation purposes.

## Best Practices

- **Consistency**: Maintain consistent ROI sizes and shapes for similar chairs and desk areas.
- **Precision**: Take time to create precise boundaries that follow the actual furniture outlines.
- **Separation**: Ensure minimal overlap between adjacent seats' ROIs to reduce detection ambiguity.
- **Documentation**: Keep notes on any special considerations for particular seats (e.g., "seat_3 is partially obscured by a column").
- **Versioning**: Maintain version control of your configuration files, especially if annotations are refined over time.

## Troubleshooting

- **Irregular Furniture**: For non-standard chairs or desk arrangements, prioritize covering the areas where a person would actually be detected rather than strictly following furniture boundaries.
- **Occlusions**: If parts of seats are permanently occluded (e.g., by columns or walls), annotate only the visible portions and note these limitations.
- **Coordinate System Issues**: Ensure the coordinate system used in annotations matches the one used in the detection system (typically top-left origin with x increasing rightward and y increasing downward).

## Next Steps

After completing the ROI annotations:

1. Integrate the configuration file into the seat occupancy detection system.
2. Test the system using the defined ROIs on various test images and videos.
3. Fine-tune ROI boundaries if detection accuracy issues are observed.
4. Document any changes made to the original annotations for future reference.

This annotation process provides the spatial foundation for our dual-criteria detection approach, enabling accurate seat occupancy monitoring through both background subtraction and person detection methods.
