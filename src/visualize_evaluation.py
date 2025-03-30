import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Set styling for plots
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Custom colors
DESK_COLOR = "#1f77b4"  # Blue
CHAIR_COLOR = "#ff7f0e"  # Orange
SPACE_COLOR = "#2ca02c"  # Green

def parse_accuracy_metrics(file_path):
    """Parse the evaluation results to extract only accuracy metrics."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Dictionary to store all metrics
    metrics = {
        'space_accuracy': {},
        'chair_accuracy': {},
        'desk_accuracy': {}
    }
    
    # Extract space-level accuracy
    space_pattern = r"== (space\d+) ==\n  Accuracy: ([\d\.]+)"
    space_matches = re.findall(space_pattern, content)
    
    print(f"Found {len(space_matches)} space matches")
    for match in space_matches:
        space_id, accuracy = match
        metrics['space_accuracy'][space_id] = float(accuracy)
    
    # More robust furniture-level accuracy pattern
    # This handles variations in the whitespace and format
    chair_pattern = r"== (space\d+) ==[\s\S]*?= chair =[\s\S]*?Frame-by-frame Accuracy: ([\d\.]+)"
    chair_matches = re.findall(chair_pattern, content)
    
    print(f"Found {len(chair_matches)} chair matches")
    for match in chair_matches:
        space_id, accuracy = match
        if accuracy != "No":
            metrics['chair_accuracy'][space_id] = float(accuracy)
    
    desk_pattern = r"== (space\d+) ==[\s\S]*?= desk =[\s\S]*?Frame-by-frame Accuracy: ([\d\.]+)"
    desk_matches = re.findall(desk_pattern, content)
    
    print(f"Found {len(desk_matches)} desk matches")
    for match in desk_matches:
        space_id, accuracy = match
        if accuracy != "No":
            metrics['desk_accuracy'][space_id] = float(accuracy)
    
    # Print what we found for debugging
    print("Chair accuracies:", metrics['chair_accuracy'])
    print("Desk accuracies:", metrics['desk_accuracy'])
    
    # Extract overall space accuracy
    overall_pattern = r"Average Space Accuracy: ([\d\.]+)"
    overall_match = re.search(overall_pattern, content)
    
    if overall_match:
        metrics['overall_space_accuracy'] = float(overall_match.group(1))
    
    return metrics

def plot_accuracy_by_type(metrics, output_path):
    """Plot the average accuracy for each detection type."""
    # Calculate average accuracies
    chair_values = list(metrics['chair_accuracy'].values())
    desk_values = list(metrics['desk_accuracy'].values())
    
    # Print for debugging
    print(f"Chair values for average: {chair_values}")
    print(f"Desk values for average: {desk_values}")
    
    chair_avg = np.mean(chair_values) if chair_values else 0
    desk_avg = np.mean(desk_values) if desk_values else 0
    space_avg = metrics['overall_space_accuracy']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    categories = ['Space', 'Desk', 'Chair']
    values = [space_avg, desk_avg, chair_avg]
    
    # Print for confirmation
    print(f"Plotting values: Space={space_avg}, Desk={desk_avg}, Chair={chair_avg}")
    
    colors = [SPACE_COLOR, DESK_COLOR, CHAIR_COLOR]
    
    bars = plt.bar(categories, values, color=colors)
    
    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.title('Average Accuracy by Detection Type')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_accuracy_by_space(metrics, output_path):
    """Plot the accuracy for each space and furniture type."""
    # Get list of spaces
    spaces = sorted(metrics['space_accuracy'].keys())
    
    # Collect accuracies for each space
    space_accuracies = []
    chair_accuracies = []
    desk_accuracies = []
    
    for space in spaces:
        space_accuracies.append(metrics['space_accuracy'][space])
        
        # Some spaces might not have chair or desk metrics
        chair_accuracies.append(metrics['chair_accuracy'].get(space, 0))
        desk_accuracies.append(metrics['desk_accuracy'].get(space, 0))
    
    # Print for debugging
    print("Space accuracies to plot:", space_accuracies)
    print("Chair accuracies to plot:", chair_accuracies)
    print("Desk accuracies to plot:", desk_accuracies)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(spaces))
    width = 0.25
    
    plt.bar(x - width, chair_accuracies, width, label='Chair', color=CHAIR_COLOR)
    plt.bar(x, desk_accuracies, width, label='Desk', color=DESK_COLOR)
    plt.bar(x + width, space_accuracies, width, label='Space', color=SPACE_COLOR)
    
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Space and Detection Type')
    plt.xticks(x, spaces)
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Add a horizontal line at 0.87 (target threshold)
    plt.axhline(y=0.87, color='gray', linestyle='--', alpha=0.7)
    plt.text(len(spaces)-1, 0.88, 'Target threshold (0.87)', ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    # Input and output paths
    evaluation_file = "evaluation_results_6.txt"
    output_dir = "accuracy_visuals"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parse the evaluation results
    metrics = parse_accuracy_metrics(evaluation_file)
    
    # Create the visualizations
    plot_accuracy_by_type(metrics, f"{output_dir}/accuracy_by_type.png")
    plot_accuracy_by_space(metrics, f"{output_dir}/accuracy_by_space.png")
    
    print(f"Accuracy charts created successfully in the '{output_dir}' directory")

if __name__ == "__main__":
    main()