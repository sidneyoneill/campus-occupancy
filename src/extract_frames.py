import cv2
import os

# Define paths
video_path = "data/videos/video_1.MOV"  # replace with your actual file name
output_dir = "data/images/frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate:", fps)

frame_count = 0
screenshot_interval = int(fps)  # Save one frame per second

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame every 'screenshot_interval' frames
    if frame_count % screenshot_interval == 0:
        output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")

    frame_count += 1

cap.release()
print("Done extracting frames.")
