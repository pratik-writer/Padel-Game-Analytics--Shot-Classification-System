import cv2
from ultralytics import YOLO
import mediapipe as mp

# 1. Initialize Models
model = YOLO('yolov8n.pt') # Use nano for speed, scale up to 's' or 'm' if needed
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 2. Setup Video I/O
cap = cv2.VideoCapture('input.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

shot_data = [] # To store your JSON/CSV data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Get timestamp

    # 3. Track objects (Players, Racket, Ball)
    # class 0 is person, 38 is tennis racket, 32 is sports ball in COCO
    results = model.track(frame, classes=[0, 32, 38], persist=True)
    
    for box in results[0].boxes:
        if int(box.cls) == 0: # If it's a person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            player_crop = frame[y1:y2, x1:x2] # Crop player for pose estimation
            
            # 4. Pose Estimation on the player
            pose_results = pose.process(cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB))
            
            if pose_results.pose_landmarks:
                # 5. Add your heuristic logic here to classify the shot
                # e.g., if wrist_y < head_y and motion detected -> 'Smash'
                shot_type = "Forehand" # Placeholder
                
                # Append to structured output
                shot_data.append({
                    "timestamp": current_time,
                    "player_id": int(box.id[0]) if box.id else None,
                    "shot_type": shot_type
                })
                
                # Draw on frame for demo video
                cv2.putText(frame, shot_type, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write frame to output video here...

cap.release()
# Finally, export shot_data to Pandas -> CSV/JSON