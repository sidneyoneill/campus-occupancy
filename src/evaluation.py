import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_frame_occupation_array(annotations, total_frames, space_id, furniture_type):
    """Create an array of occupation status for each frame."""
    # Initialize array with None values
    occupation = [None] * (total_frames + 1)  # +1 because frames are 1-indexed
    
    # If the space or furniture type doesn't exist in annotations, return array of Nones
    if space_id not in annotations or furniture_type not in annotations[space_id]:
        return occupation
    
    # Fill occupation array based on annotation intervals
    for interval in annotations[space_id][furniture_type]:
        start = interval["start_frame"]
        end = interval["end_frame"]
        occupied = interval["occupied"]
        
        # Make sure we don't exceed array bounds
        end = min(end, total_frames)
        
        # Fill the range with the occupation status
        for frame in range(start, end + 1):
            occupation[frame] = occupied
            
    return occupation

def calculate_metrics(gt_array, pred_array, threshold=0.5):
    """Calculate performance metrics between ground truth and predictions."""
    # Filter out frames where either array has None values
    valid_indices = [i for i in range(len(gt_array)) 
                     if gt_array[i] is not None and pred_array[i] is not None]
    
    if not valid_indices:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "confusion_matrix": None
        }
    
    gt_filtered = [gt_array[i] for i in valid_indices]
    pred_filtered = [pred_array[i] for i in valid_indices]
    
    # Convert boolean values to integers for metrics calculation
    gt_int = [1 if x else 0 for x in gt_filtered]
    pred_int = [1 if x else 0 for x in pred_filtered]
    
    # Calculate metrics
    accuracy = accuracy_score(gt_int, pred_int)
    precision, recall, f1, _ = precision_recall_fscore_support(gt_int, pred_int, average='binary')
    cm = confusion_matrix(gt_int, pred_int)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def calculate_temporal_iou(gt_intervals, pred_intervals):
    """Calculate temporal IoU (Intersection over Union) for occupancy intervals."""
    total_intersection = 0
    total_union = 0
    
    # Create frame-by-frame occupancy arrays
    max_frame = 0
    for interval in gt_intervals + pred_intervals:
        max_frame = max(max_frame, interval["end_frame"])
    
    gt_frames = [False] * (max_frame + 1)
    pred_frames = [False] * (max_frame + 1)
    
    # Fill ground truth frames
    for interval in gt_intervals:
        if interval["occupied"]:
            for frame in range(interval["start_frame"], interval["end_frame"] + 1):
                if frame < len(gt_frames):
                    gt_frames[frame] = True
    
    # Fill prediction frames
    for interval in pred_intervals:
        if interval["occupied"]:
            for frame in range(interval["start_frame"], interval["end_frame"] + 1):
                if frame < len(pred_frames):
                    pred_frames[frame] = True
    
    # Calculate intersection and union
    for i in range(len(gt_frames)):
        if gt_frames[i] and pred_frames[i]:
            total_intersection += 1
        if gt_frames[i] or pred_frames[i]:
            total_union += 1
    
    return total_intersection / total_union if total_union > 0 else 0

def calculate_event_detection_metrics(gt_intervals, pred_intervals):
    """Calculate event-based detection metrics (for occupied periods)."""
    # Only consider occupied intervals
    gt_occupied = [interval for interval in gt_intervals if interval["occupied"]]
    pred_occupied = [interval for interval in pred_intervals if interval["occupied"]]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Define overlap threshold for counting as a true positive
    overlap_threshold = 0.5
    
    # For each ground truth interval, check if it overlaps with any prediction
    for gt in gt_occupied:
        gt_start = gt["start_frame"]
        gt_end = gt["end_frame"]
        gt_duration = gt_end - gt_start + 1
        
        max_overlap = 0
        best_pred = None
        
        for pred in pred_occupied:
            pred_start = pred["start_frame"]
            pred_end = pred["end_frame"]
            
            # Calculate overlap
            overlap_start = max(gt_start, pred_start)
            overlap_end = min(gt_end, pred_end)
            overlap_duration = max(0, overlap_end - overlap_start + 1)
            
            overlap_ratio = overlap_duration / gt_duration
            
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_pred = pred
        
        if max_overlap >= overlap_threshold:
            true_positives += 1
            # Remove the matched prediction to prevent double counting
            if best_pred in pred_occupied:
                pred_occupied.remove(best_pred)
        else:
            false_negatives += 1
    
    # Any remaining predictions are false positives
    false_positives = len(pred_occupied)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def calculate_transition_detection_metrics(gt_intervals, pred_intervals):
    """Calculate metrics for transition detection (how well the system detects changes in state)."""
    # Extract transition points (start and end of each interval)
    gt_transitions = []
    pred_transitions = []
    
    for i in range(len(gt_intervals)):
        gt_transitions.append(gt_intervals[i]["start_frame"])
        if i < len(gt_intervals) - 1 and gt_intervals[i]["end_frame"] + 1 != gt_intervals[i+1]["start_frame"]:
            gt_transitions.append(gt_intervals[i]["end_frame"])
    
    for i in range(len(pred_intervals)):
        pred_transitions.append(pred_intervals[i]["start_frame"])
        if i < len(pred_intervals) - 1 and pred_intervals[i]["end_frame"] + 1 != pred_intervals[i+1]["start_frame"]:
            pred_transitions.append(pred_intervals[i]["end_frame"])
    
    # Define a tolerance window (in frames) for transition detection
    tolerance = 30  # 1 second at 30fps
    
    matched_gt = []
    matched_pred = []
    
    # Count matches within tolerance
    for gt_trans in gt_transitions:
        for pred_trans in pred_transitions:
            if abs(gt_trans - pred_trans) <= tolerance and pred_trans not in matched_pred:
                matched_gt.append(gt_trans)
                matched_pred.append(pred_trans)
                break
    
    # Calculate metrics
    true_positives = len(matched_gt)
    false_positives = len(pred_transitions) - true_positives
    false_negatives = len(gt_transitions) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def calculate_completely_wrong_predictions(gt_array, pred_array):
    """Calculate instances where predictions are completely wrong (occupied vs. unoccupied)."""
    # Filter out frames where either array has None values
    valid_indices = [i for i in range(len(gt_array)) 
                     if gt_array[i] is not None and pred_array[i] is not None]
    
    if not valid_indices:
        return 0, 0
    
    gt_filtered = [gt_array[i] for i in valid_indices]
    pred_filtered = [pred_array[i] for i in valid_indices]
    
    # Count frames where prediction is opposite of ground truth
    completely_wrong = sum(1 for i in range(len(gt_filtered)) if gt_filtered[i] != pred_filtered[i])
    
    # Calculate percentage
    percentage_wrong = completely_wrong / len(valid_indices) if valid_indices else 0
    
    return completely_wrong, percentage_wrong

def evaluate_space_level_occupancy(ground_truth, detection_results, total_frames, space_id):
    """Evaluate if a space (desk or chair) is correctly identified as occupied/unoccupied."""
    # Create arrays to track if any furniture in the space is occupied
    gt_space_occupied = [False] * (total_frames + 1)
    pred_space_occupied = [False] * (total_frames + 1)
    
    # Fill ground truth space occupancy (if either desk or chair is occupied, space is occupied)
    for furniture_type in ["chair", "desk"]:
        if (space_id in ground_truth and furniture_type in ground_truth[space_id]):
            for interval in ground_truth[space_id][furniture_type]:
                if interval["occupied"]:
                    start = interval["start_frame"]
                    end = min(interval["end_frame"], total_frames)
                    for frame in range(start, end + 1):
                        gt_space_occupied[frame] = True
    
    # Fill prediction space occupancy
    for furniture_type in ["chair", "desk"]:
        if (space_id in detection_results and furniture_type in detection_results[space_id]):
            for interval in detection_results[space_id][furniture_type]:
                if interval["occupied"]:
                    start = interval["start_frame"]
                    end = min(interval["end_frame"], total_frames)
                    for frame in range(start, end + 1):
                        pred_space_occupied[frame] = True
    
    # Convert to binary arrays for metrics calculation (None values become False)
    gt_binary = [1 if x else 0 for x in gt_space_occupied]
    pred_binary = [1 if x else 0 for x in pred_space_occupied]
    
    # Calculate metrics
    accuracy = accuracy_score(gt_binary, pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_binary, pred_binary, average='binary', zero_division=0)
    cm = confusion_matrix(gt_binary, pred_binary)
    
    # Calculate completely wrong predictions
    wrong_count, wrong_percentage = calculate_completely_wrong_predictions(
        gt_space_occupied, pred_space_occupied)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "wrong_count": wrong_count,
        "wrong_percentage": wrong_percentage
    }

def plot_confusion_matrices(all_metrics):
    """Plot confusion matrices for all spaces and furniture types."""
    plt.figure(figsize=(15, 10))
    
    # Count how many valid confusion matrices we have
    valid_cms = 0
    for space_metrics in all_metrics.values():
        for furniture_metrics in space_metrics.values():
            if furniture_metrics["frame_metrics"]["confusion_matrix"] is not None:
                valid_cms += 1
    
    if valid_cms == 0:
        return
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(valid_cms)))
    plot_idx = 1
    
    for space_id, space_metrics in all_metrics.items():
        for furniture_type, furniture_metrics in space_metrics.items():
            cm = furniture_metrics["frame_metrics"]["confusion_matrix"]
            if cm is not None:
                plt.subplot(grid_size, grid_size, plot_idx)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Unoccupied', 'Occupied'],
                            yticklabels=['Unoccupied', 'Occupied'])
                plt.title(f"{space_id} - {furniture_type}")
                plt.ylabel('Ground Truth')
                plt.xlabel('Prediction')
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def identify_missed_occupancy_events(gt_intervals, pred_intervals, overlap_threshold=0.1, min_duration_frames=0):
    """
    Identify ground truth occupancy events that were completely missed or barely detected.
    
    Args:
        gt_intervals: Ground truth intervals
        pred_intervals: Predicted intervals
        overlap_threshold: Minimum overlap ratio to not consider an event as "missed"
        min_duration_frames: Minimum duration (in frames) for an event to be considered significant
        
    Returns:
        List of missed events with their details
    """
    # Only consider occupied intervals
    gt_occupied = [interval for interval in gt_intervals if interval["occupied"]]
    pred_occupied = [interval for interval in pred_intervals if interval["occupied"]]
    
    missed_events = []
    
    # For each ground truth occupied interval
    for gt in gt_occupied:
        gt_start = gt["start_frame"]
        gt_end = gt["end_frame"]
        gt_duration = gt_end - gt_start + 1
        
        # Skip events shorter than the minimum duration
        if gt_duration < min_duration_frames:
            continue
            
        max_overlap = 0
        best_pred = None
        
        for pred in pred_occupied:
            pred_start = pred["start_frame"]
            pred_end = pred["end_frame"]
            
            # Calculate overlap
            overlap_start = max(gt_start, pred_start)
            overlap_end = min(gt_end, pred_end)
            overlap_duration = max(0, overlap_end - overlap_start + 1)
            
            overlap_ratio = overlap_duration / gt_duration
            
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_pred = pred
        
        # If the overlap is below threshold, consider the event as missed
        if max_overlap < overlap_threshold:
            missed_events.append({
                "event": gt,
                "overlap_ratio": max_overlap,
                "duration": gt_duration,
                "duration_seconds": gt_duration / 30.0  # Assuming 30fps
            })
    
    return missed_events

def identify_false_occupancy_events(gt_intervals, pred_intervals, overlap_threshold=0.1, min_duration_frames=0):
    """
    Identify predicted occupancy events that are mostly or completely false.
    
    Args:
        gt_intervals: Ground truth intervals
        pred_intervals: Predicted intervals
        overlap_threshold: Maximum overlap ratio to consider an event as "false"
        min_duration_frames: Minimum duration (in frames) for an event to be considered significant
        
    Returns:
        List of false events with their details
    """
    # Only consider occupied intervals
    gt_occupied = [interval for interval in gt_intervals if interval["occupied"]]
    pred_occupied = [interval for interval in pred_intervals if interval["occupied"]]
    
    false_events = []
    
    # For each predicted occupied interval
    for pred in pred_occupied:
        pred_start = pred["start_frame"]
        pred_end = pred["end_frame"]
        pred_duration = pred_end - pred_start + 1
        
        # Skip events shorter than the minimum duration
        if pred_duration < min_duration_frames:
            continue
            
        max_overlap = 0
        best_gt = None
        
        for gt in gt_occupied:
            gt_start = gt["start_frame"]
            gt_end = gt["end_frame"]
            
            # Calculate overlap
            overlap_start = max(gt_start, pred_start)
            overlap_end = min(gt_end, pred_end)
            overlap_duration = max(0, overlap_end - overlap_start + 1)
            
            overlap_ratio = overlap_duration / pred_duration
            
            if overlap_ratio > max_overlap:
                max_overlap = overlap_ratio
                best_gt = gt
        
        # If the overlap is below threshold, consider the event as false
        if max_overlap < overlap_threshold:
            false_events.append({
                "event": pred,
                "overlap_ratio": max_overlap,
                "duration": pred_duration,
                "duration_seconds": pred_duration / 30.0  # Assuming 30fps
            })
    
    return false_events

def summarize_occupancy_events(gt_intervals, pred_intervals, overlap_threshold=0.1, min_duration_frames=0):
    """
    Summarize statistics about occupancy events in ground truth and predictions.
    
    Args:
        gt_intervals: Ground truth intervals
        pred_intervals: Predicted intervals
        overlap_threshold: Threshold for considering events as missed/false
        min_duration_frames: Minimum duration (in frames) for an event to be considered significant
        
    Returns:
        Dictionary with event summary statistics
    """
    # Only consider occupied intervals
    gt_occupied = [interval for interval in gt_intervals if interval["occupied"]]
    pred_occupied = [interval for interval in pred_intervals if interval["occupied"]]
    
    # Filter by duration if specified
    if min_duration_frames > 0:
        gt_occupied = [interval for interval in gt_occupied 
                       if (interval["end_frame"] - interval["start_frame"] + 1) >= min_duration_frames]
        pred_occupied = [interval for interval in pred_occupied 
                         if (interval["end_frame"] - interval["start_frame"] + 1) >= min_duration_frames]
    
    # Count events
    n_gt_events = len(gt_occupied)
    n_pred_events = len(pred_occupied)
    
    # Calculate event durations
    gt_durations = [(event["end_frame"] - event["start_frame"] + 1) for event in gt_occupied]
    pred_durations = [(event["end_frame"] - event["start_frame"] + 1) for event in pred_occupied]
    
    # Calculate statistics about event durations
    avg_gt_duration = np.mean(gt_durations) if gt_durations else 0
    avg_pred_duration = np.mean(pred_durations) if pred_durations else 0
    
    # Find missed and false events with duration filtering
    missed_events = identify_missed_occupancy_events(gt_intervals, pred_intervals, overlap_threshold, min_duration_frames)
    false_events = identify_false_occupancy_events(gt_intervals, pred_intervals, overlap_threshold, min_duration_frames)
    
    # Calculate overlap metrics for matched events
    matched_gt_events = [event for event in gt_occupied if event not in [e["event"] for e in missed_events]]
    matched_pred_events = [event for event in pred_occupied if event not in [e["event"] for e in false_events]]
    
    return {
        "gt_events": n_gt_events,
        "pred_events": n_pred_events,
        "avg_gt_duration_frames": avg_gt_duration,
        "avg_gt_duration_seconds": avg_gt_duration / 30.0,  # Assuming 30fps
        "avg_pred_duration_frames": avg_pred_duration,
        "avg_pred_duration_seconds": avg_pred_duration / 30.0,  # Assuming 30fps
        "missed_events": missed_events,
        "false_events": false_events,
        "n_missed_events": len(missed_events),
        "n_false_events": len(false_events),
        "percent_missed": len(missed_events) / n_gt_events if n_gt_events > 0 else 0,
        "percent_false": len(false_events) / n_pred_events if n_pred_events > 0 else 0
    }

def main():
    # Load the ground truth and detection result data
    ground_truth_path = "annotations/ground_truth_2.json"
    detection_results_path = "output/detection_results_6.json"
    
    # Define significance thresholds (configurable)
    min_event_duration_frames = 200  # 1 second at 30fps - events shorter than this are ignored
    min_event_duration_seconds = min_event_duration_frames / 30.0
    overlap_threshold = 0.1  # 10% overlap threshold for missed/false events
    
    # Parse command line arguments if needed
    parser = argparse.ArgumentParser(description='Evaluate occupancy detection performance')
    parser.add_argument('--gt', type=str, default=ground_truth_path, help='Path to ground truth JSON file')
    parser.add_argument('--pred', type=str, default=detection_results_path, help='Path to prediction JSON file')
    parser.add_argument('--min-duration', type=float, default=min_event_duration_seconds, 
                        help='Minimum duration (in seconds) for an event to be considered significant')
    parser.add_argument('--overlap-threshold', type=float, default=overlap_threshold,
                        help='Overlap threshold for missed/false events (0.0-1.0)')
    args = parser.parse_args()
    
    # Update parameters from command line arguments if provided
    ground_truth_path = args.gt
    detection_results_path = args.pred
    min_event_duration_seconds = args.min_duration
    min_event_duration_frames = int(min_event_duration_seconds * 30.0)  # Convert to frames
    overlap_threshold = args.overlap_threshold
    
    try:
        ground_truth = load_json_data(ground_truth_path)
        detection_results = load_json_data(detection_results_path)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON data: {e}")
        # The detection_results file appears to be incomplete based on the provided content
        # Let's implement a fallback to handle this
        print("Detection results file may be incomplete. Using available data.")
        with open(detection_results_path, 'r') as f:
            content = f.read()
        # Complete the JSON if it's truncated
        if not content.strip().endswith('}'):
            content += '}}}}}'  # Add missing closing brackets as needed
            detection_results = json.loads(content)
    
    # Print evaluation parameters
    print("\n===== EVALUATION PARAMETERS =====")
    print(f"Minimum significant event duration: {min_event_duration_seconds:.1f} seconds ({min_event_duration_frames} frames)")
    print(f"Overlap threshold for missed/false events: {overlap_threshold:.1%}\n")
    
    # Use the total frames from ground truth
    total_frames = ground_truth["total_frames"]
    
    # Initialize results dictionary
    all_metrics = {}
    space_level_metrics = {}
    completely_wrong_counts = {}
    
    # Analyze each space and furniture type
    for space_id in ground_truth["annotations"]:
        all_metrics[space_id] = {}
        completely_wrong_counts[space_id] = {}
        
        for furniture_type in ["chair", "desk"]:
            # Create frame-by-frame occupation arrays
            gt_occupation = create_frame_occupation_array(
                ground_truth["annotations"], total_frames, space_id, furniture_type)
            
            # Check if this space/furniture exists in detection results
            if (detection_results.get("annotations") and 
                space_id in detection_results["annotations"] and 
                furniture_type in detection_results["annotations"][space_id]):
                pred_occupation = create_frame_occupation_array(
                    detection_results["annotations"], total_frames, space_id, furniture_type)
            else:
                # If not present in detection results, create array of None values
                pred_occupation = [None] * (total_frames + 1)
            
            # Calculate completely wrong predictions
            wrong_count, wrong_percentage = calculate_completely_wrong_predictions(
                gt_occupation, pred_occupation)
            completely_wrong_counts[space_id][furniture_type] = {
                "count": wrong_count,
                "percentage": wrong_percentage
            }
            
            # Calculate frame-by-frame metrics
            frame_metrics = calculate_metrics(gt_occupation, pred_occupation)
            
            # Calculate temporal IoU
            temporal_iou = 0
            if (space_id in ground_truth["annotations"] and 
                furniture_type in ground_truth["annotations"][space_id] and
                space_id in detection_results.get("annotations", {}) and 
                furniture_type in detection_results.get("annotations", {}).get(space_id, {})):
                temporal_iou = calculate_temporal_iou(
                    ground_truth["annotations"][space_id][furniture_type],
                    detection_results["annotations"][space_id][furniture_type]
                )
            
            # Calculate event detection metrics
            event_metrics = {}
            event_summary = {}
            if (space_id in ground_truth["annotations"] and 
                furniture_type in ground_truth["annotations"][space_id] and
                space_id in detection_results.get("annotations", {}) and 
                furniture_type in detection_results.get("annotations", {}).get(space_id, {})):
                event_metrics = calculate_event_detection_metrics(
                    ground_truth["annotations"][space_id][furniture_type],
                    detection_results["annotations"][space_id][furniture_type]
                )
                # Add event summary with missed and false events, using configured thresholds
                event_summary = summarize_occupancy_events(
                    ground_truth["annotations"][space_id][furniture_type],
                    detection_results["annotations"][space_id][furniture_type],
                    overlap_threshold,
                    min_event_duration_frames
                )
            
            # Calculate transition detection metrics
            transition_metrics = {}
            if (space_id in ground_truth["annotations"] and 
                furniture_type in ground_truth["annotations"][space_id] and
                space_id in detection_results.get("annotations", {}) and 
                furniture_type in detection_results.get("annotations", {}).get(space_id, {})):
                transition_metrics = calculate_transition_detection_metrics(
                    ground_truth["annotations"][space_id][furniture_type],
                    detection_results["annotations"][space_id][furniture_type]
                )
            
            # Store all metrics including event summary
            all_metrics[space_id][furniture_type] = {
                "frame_metrics": frame_metrics,
                "temporal_iou": temporal_iou,
                "event_metrics": event_metrics,
                "event_summary": event_summary,
                "transition_metrics": transition_metrics
            }
        
        # Calculate space-level metrics (if either desk or chair is occupied, space is occupied)
        space_level_metrics[space_id] = evaluate_space_level_occupancy(
            ground_truth["annotations"], 
            detection_results.get("annotations", {}),
            total_frames,
            space_id
        )
    
    # Calculate overall averages
    overall_accuracy = []
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    overall_iou = []
    overall_event_f1 = []
    overall_transition_f1 = []
    
    for space_metrics in all_metrics.values():
        for furniture_metrics in space_metrics.values():
            if furniture_metrics["frame_metrics"]["accuracy"] is not None:
                overall_accuracy.append(furniture_metrics["frame_metrics"]["accuracy"])
                overall_precision.append(furniture_metrics["frame_metrics"]["precision"])
                overall_recall.append(furniture_metrics["frame_metrics"]["recall"])
                overall_f1.append(furniture_metrics["frame_metrics"]["f1_score"])
            
            if furniture_metrics["temporal_iou"] > 0:
                overall_iou.append(furniture_metrics["temporal_iou"])
            
            if "f1_score" in furniture_metrics.get("event_metrics", {}):
                overall_event_f1.append(furniture_metrics["event_metrics"]["f1_score"])
            
            if "f1_score" in furniture_metrics.get("transition_metrics", {}):
                overall_transition_f1.append(furniture_metrics["transition_metrics"]["f1_score"])
    
    # Print results including missed and false events
    print("\n===== DETECTION PERFORMANCE EVALUATION =====\n")
    
    # Print per-space, per-furniture metrics
    for space_id, space_metrics in all_metrics.items():
        print(f"\n== {space_id} ==")
        
        for furniture_type, furniture_metrics in space_metrics.items():
            print(f"\n= {furniture_type} =")
            
            # Frame-by-frame metrics
            frame_metrics = furniture_metrics["frame_metrics"]
            if frame_metrics["accuracy"] is not None:
                print(f"Frame-by-frame Accuracy: {frame_metrics['accuracy']:.4f}")
                print(f"Frame-by-frame Precision: {frame_metrics['precision']:.4f}")
                print(f"Frame-by-frame Recall: {frame_metrics['recall']:.4f}")
                print(f"Frame-by-frame F1 Score: {frame_metrics['f1_score']:.4f}")
            else:
                print("No valid frame-by-frame metrics available")
            
            # Temporal IoU
            print(f"Temporal IoU: {furniture_metrics['temporal_iou']:.4f}")
            
            # Event detection metrics
            if furniture_metrics.get("event_metrics"):
                event_metrics = furniture_metrics["event_metrics"]
                print("\nEvent Detection Metrics:")
                print(f"  True Positives: {event_metrics.get('true_positives', 0)}")
                print(f"  False Positives: {event_metrics.get('false_positives', 0)}")
                print(f"  False Negatives: {event_metrics.get('false_negatives', 0)}")
                print(f"  Precision: {event_metrics.get('precision', 0):.4f}")
                print(f"  Recall: {event_metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {event_metrics.get('f1_score', 0):.4f}")
            
            # Transition detection metrics
            if furniture_metrics.get("transition_metrics"):
                transition_metrics = furniture_metrics["transition_metrics"]
                print("\nTransition Detection Metrics:")
                print(f"  True Positives: {transition_metrics.get('true_positives', 0)}")
                print(f"  False Positives: {transition_metrics.get('false_positives', 0)}")
                print(f"  False Negatives: {transition_metrics.get('false_negatives', 0)}")
                print(f"  Precision: {transition_metrics.get('precision', 0):.4f}")
                print(f"  Recall: {transition_metrics.get('recall', 0):.4f}")
                print(f"  F1 Score: {transition_metrics.get('f1_score', 0):.4f}")
            
            # Print event summary information with duration threshold information
            if "event_summary" in furniture_metrics and furniture_metrics["event_summary"]:
                summary = furniture_metrics["event_summary"]
                print("\nOccupancy Event Summary:")
                print(f"  GT Events (>= {min_event_duration_seconds:.1f}s): {summary.get('gt_events', 0)}")
                print(f"  Pred Events (>= {min_event_duration_seconds:.1f}s): {summary.get('pred_events', 0)}")
                print(f"  Avg GT Duration: {summary.get('avg_gt_duration_seconds', 0):.2f}s, " 
                      f"Avg Pred Duration: {summary.get('avg_pred_duration_seconds', 0):.2f}s")
                print(f"  Significant Missed Events: {summary.get('n_missed_events', 0)} "
                      f"({summary.get('percent_missed', 0):.1%} of GT events)")
                print(f"  Significant False Events: {summary.get('n_false_events', 0)} "
                      f"({summary.get('percent_false', 0):.1%} of predicted events)")
                
                # Print details of missed events
                if summary.get('missed_events', []):
                    print(f"\n  Details of Missed Occupancy Events (>= {min_event_duration_seconds:.1f}s):")
                    for i, event in enumerate(summary['missed_events']):
                        duration_sec = event['duration_seconds']
                        print(f"    {i+1}. Frames {event['event']['start_frame']}-{event['event']['end_frame']} "
                              f"(Duration: {duration_sec:.1f}s, Overlap: {event['overlap_ratio']:.1%})")
                
                # Print details of false events
                if summary.get('false_events', []):
                    print(f"\n  Details of False Occupancy Events (>= {min_event_duration_seconds:.1f}s):")
                    for i, event in enumerate(summary['false_events']):
                        duration_sec = event['duration_seconds']
                        print(f"    {i+1}. Frames {event['event']['start_frame']}-{event['event']['end_frame']} "
                              f"(Duration: {duration_sec:.1f}s, Overlap: {event['overlap_ratio']:.1%})")
    
    # Print completely wrong predictions
    print("\n===== COMPLETELY WRONG PREDICTIONS =====")
    for space_id, furniture_metrics in completely_wrong_counts.items():
        print(f"\n== {space_id} ==")
        for furniture_type, wrong_data in furniture_metrics.items():
            print(f"  {furniture_type}: {wrong_data['count']} frames ({wrong_data['percentage']:.2%})")
    
    # Print space-level metrics
    print("\n===== SPACE-LEVEL OCCUPANCY METRICS =====")
    for space_id, metrics in space_level_metrics.items():
        print(f"\n== {space_id} ==")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Completely Wrong: {metrics['wrong_count']} frames ({metrics['wrong_percentage']:.2%})")
    
    # Print space-level averages
    print("\n===== OVERALL SPACE-LEVEL PERFORMANCE =====")
    print(f"Average Space Accuracy: {np.mean(space_level_metrics[space_id]['accuracy']):.4f} (±{np.std(space_level_metrics[space_id]['accuracy']):.4f})")
    print(f"Average Space Precision: {np.mean(space_level_metrics[space_id]['precision']):.4f} (±{np.std(space_level_metrics[space_id]['precision']):.4f})")
    print(f"Average Space Recall: {np.mean(space_level_metrics[space_id]['recall']):.4f} (±{np.std(space_level_metrics[space_id]['recall']):.4f})")
    print(f"Average Space F1 Score: {np.mean(space_level_metrics[space_id]['f1_score']):.4f} (±{np.std(space_level_metrics[space_id]['f1_score']):.4f})")
    
    # Plot space-level confusion matrices
    plt.figure(figsize=(10, 8))
    for i, (space_id, metrics) in enumerate(space_level_metrics.items(), 1):
        plt.subplot(3, 3, i)
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Unoccupied', 'Occupied'],
                    yticklabels=['Unoccupied', 'Occupied'])
        plt.title(f"{space_id} - Space Level")
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
    
    plt.tight_layout()
    plt.savefig('space_level_confusion_matrices.png')
    plt.close()
    
    print("\nSpace-level results saved to space_level_confusion_matrices.png")

    # Summarize missed event statistics across all spaces and furniture, emphasizing significance threshold
    total_gt_events = 0
    total_missed_events = 0
    total_false_events = 0
    
    for space_metrics in all_metrics.values():
        for furniture_metrics in space_metrics.values():
            if "event_summary" in furniture_metrics and furniture_metrics["event_summary"]:
                summary = furniture_metrics["event_summary"]
                total_gt_events += summary.get('gt_events', 0)
                total_missed_events += summary.get('n_missed_events', 0)
                total_false_events += summary.get('n_false_events', 0)
    
    print("\n===== OVERALL EVENT DETECTION PERFORMANCE =====")
    print(f"Events considered significant: >= {min_event_duration_seconds:.1f} seconds")
    print(f"Total significant GT occupancy events: {total_gt_events}")
    print(f"Total significant missed events: {total_missed_events} ({total_missed_events/total_gt_events:.1%} if total_gt_events > 0)")
    print(f"Total significant false events: {total_false_events}")

if __name__ == "__main__":
    main()