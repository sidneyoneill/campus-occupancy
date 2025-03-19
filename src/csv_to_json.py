import csv
import json
import os

# Define paths
csv_path = os.path.join("annotations", "chair_locations_2.csv")   # Replace with your actual CSV filename
json_path = os.path.join("annotations", "chair_locations_2.json")

seat_data = {}  # e.g., {"seat1": {"chair": [x, y, w, h], "desk": [x, y, w, h]}, ...}

with open(csv_path, mode='r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Read from your CSV columns
        label = row['label_name']               # e.g., "chair_seat_1" or "desk_area_1"
        x = int(float(row['bbox_x']))
        y = int(float(row['bbox_y']))
        w = int(float(row['bbox_width']))
        h = int(float(row['bbox_height']))

        # Example label_name: "chair_seat_1"
        parts = label.split("_")
        # parts[0] => "chair" or "desk"
        # parts[1] => "seat" or "area" (depending on how you named them)
        # parts[-1] => the seat number ("1", "2", etc.)

        # We'll assume the seat number is always at the end.
        seat_number = parts[-1]  # e.g. "1"
        roi_type = parts[0]      # "chair" or "desk"
        seat_key = f"seat{seat_number}"

        if seat_key not in seat_data:
            seat_data[seat_key] = {}

        seat_data[seat_key][roi_type] = [x, y, w, h]

# Save to JSON
os.makedirs(os.path.dirname(json_path), exist_ok=True)
with open(json_path, "w") as jsonfile:
    json.dump(seat_data, jsonfile, indent=2)

print(f"Annotations saved to {json_path}")

