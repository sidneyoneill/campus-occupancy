import cv2
import json
import os
import numpy as np

def load_annotations(annotations_path):
    """Load ROI annotations from a JSON file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)
    
def extract_frame_at_interval(video_path, seconds_interval):
    """
    Extract frames from the video at every 'seconds_interval' seconds.
    Returns a list of tuples (frame, frame_index).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * seconds_interval)
    extracted_frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            extracted_frames.append((frame.copy(), frame_index))
        frame_index += 1

    cap.release()
    return extracted_frames

def create_baseline_images(empty_frame, annotations):
    """
    Create baseline images by cropping the desk ROI from an empty reference frame.
    Returns a dictionary mapping seat_id -> baseline ROI.
    """
    baseline_images = {}
    for seat_id, rois in annotations.items():
        # We expect only a "desk" ROI in the new annotations.
        if "desk" in rois:
            x, y, width, height = rois["desk"]
            roi = empty_frame[y:y+height, x:x+width]
            baseline_images[seat_id] = roi
    return baseline_images

def create_gaussian_mask(shape, sigma_factor=0.5):
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

def is_desk_occupied_proportion_weighted(current_frame, baseline_images, seat_id, annotations,
                                         diff_threshold=30, proportion_threshold=0.05):
    """
    Determine occupancy using a Gaussian-weighted proportion of changed pixels.
    Returns a tuple (occupied (bool), confidence (float)).
    """
    # Get the desk ROI from the annotations.
    x, y, width, height = annotations[seat_id]["desk"]
    current_roi = current_frame[y:y+height, x:x+width]
    baseline_roi = baseline_images[seat_id]
    
    if current_roi.shape != baseline_roi.shape:
        current_roi = cv2.resize(current_roi, (baseline_roi.shape[1], baseline_roi.shape[0]))
    
    current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY) if len(current_roi.shape) == 3 else current_roi
    baseline_gray = cv2.cvtColor(baseline_roi, cv2.COLOR_BGR2GRAY) if len(baseline_roi.shape) == 3 else baseline_roi
    
    diff = cv2.absdiff(current_gray, baseline_gray)
    gaussian_mask = create_gaussian_mask(diff.shape, sigma_factor=0.5)
    weighted_diff = diff * gaussian_mask
    binary_mask = (weighted_diff > diff_threshold).astype(np.uint8)
    weighted_proportion = np.sum(binary_mask * gaussian_mask) / np.sum(gaussian_mask)
    confidence = min(100, weighted_proportion * 100)
    occupied = weighted_proportion > proportion_threshold
    return occupied, confidence

def visualize_occupancy(frame, annotations, occupied_data):
    """
    Draw bounding boxes and labels for each desk ROI on the frame.
    """
    vis_frame = frame.copy()
    colors = {'occupied': (0, 0, 255), 'empty': (0, 255, 0)}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for seat_id, rois in annotations.items():
        # Since we only have desk ROIs, we can directly access them.
        x, y, width, height = rois["desk"]
        occ, conf = False, 0
        if seat_id in occupied_data:
            occ, conf = True, occupied_data[seat_id]
        status = "Occupied" if occ else "Empty"
        color_box = colors['occupied'] if occ else colors['empty']
        cv2.rectangle(vis_frame, (x, y), (x + width, y + height), color_box, 2)
        label = f"{seat_id}: desk ({status} {conf:.1f}%)"
        cv2.putText(vis_frame, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    return vis_frame

def main():
    baseline_path = "frame_00050.jpg"
    video_path = "birdseyevid.MOV"
    # Use the new annotations file with only desk ROIs.
    annotations_path = os.path.join("annotations", "annotations_only_desk.json")
    
    baseline_img = cv2.imread(baseline_path)
    if baseline_img is None:
        print(f"Error: Could not load baseline image from {baseline_path}")
        return
    
    annotations = load_annotations(annotations_path)
    baseline_images = create_baseline_images(baseline_img, annotations)
    
    frames = extract_frame_at_interval(video_path, seconds_interval=5)
    if not frames:
        print("No frames extracted from the video.")
        return
    
    output_dir = "occupancy_results_weighted_desks"
    os.makedirs(output_dir, exist_ok=True)
    
    for frame, frame_number in frames:
        occupied_data = {}
        for seat_id in annotations:
            occupied, confidence = is_desk_occupied_proportion_weighted(
                frame, baseline_images, seat_id, annotations
            )
            print(f"Frame {frame_number}: {seat_id} desk: {'Occupied' if occupied else 'Empty'} (Confidence: {confidence:.1f}%)")
            if occupied:
                occupied_data[seat_id] = confidence
        vis_frame = visualize_occupancy(frame, annotations, occupied_data)
        output_path = os.path.join(output_dir, f"frame_{frame_number:05d}_vis.jpg")
        cv2.imwrite(output_path, vis_frame)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
