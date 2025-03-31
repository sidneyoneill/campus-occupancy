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
    
    # More specific pattern for furniture-level accuracy that handles exact framing
    # This looks for Frame-by-frame Accuracy: followed by a decimal number
    for space_id in metrics['space_accuracy'].keys():
        # Find sections for each space
        space_section_pattern = f"== {space_id} ==([\\s\\S]*?)(?:== space|$)"
        space_section_match = re.search(space_section_pattern, content)
        
        if space_section_match:
            space_content = space_section_match.group(1)
            
            # Find chair and desk subsections
            chair_section_pattern = r"= chair =([\s\S]*?)(?:= desk =|$)"
            chair_section_match = re.search(chair_section_pattern, space_content)
            
            desk_section_pattern = r"= desk =([\s\S]*?)(?:$)"
            desk_section_match = re.search(desk_section_pattern, space_content)
            
            # Extract chair accuracy
            if chair_section_match:
                chair_content = chair_section_match.group(1)
                chair_accuracy_pattern = r"Frame-by-frame Accuracy: ([\d\.]+)"
                chair_accuracy_match = re.search(chair_accuracy_pattern, chair_content)
                
                if chair_accuracy_match and "No valid" not in chair_content[:50]:
                    metrics['chair_accuracy'][space_id] = float(chair_accuracy_match.group(1))
            
            # Extract desk accuracy
            if desk_section_match:
                desk_content = desk_section_match.group(1)
                desk_accuracy_pattern = r"Frame-by-frame Accuracy: ([\d\.]+)"
                desk_accuracy_match = re.search(desk_accuracy_pattern, desk_content)
                
                if desk_accuracy_match and "No valid" not in desk_content[:50]:
                    metrics['desk_accuracy'][space_id] = float(desk_accuracy_match.group(1))
    
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
    evaluation_file = "output/evaluation_results_main6_v5.txt"
    output_dir = "accuracy_visuals_main6_v5"
    
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