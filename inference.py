import cv2
import time
import csv
import os
from datetime import datetime
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
VIDEO_INPUT = 0  # ⚠️ CHANGE THIS TO YOUR EXACT VIDEO NAME!
MODEL_ONNX = "best_int8.onnx"
LOG_FILE = "road_damage_log3.csv"
SNAPSHOT_DIR = "snapshots3"

# --- 🚨 PRE-FLIGHT CRASH CHECK 🚨 ---
if not os.path.exists(VIDEO_INPUT):
    print(f"\n❌ CRITICAL ERROR: I cannot find '{VIDEO_INPUT}'!")
    print("Please make sure the video is in the exact same folder as this Python script!")
    exit()

if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

print(f"🍓 Booting up Pi 4 to process {VIDEO_INPUT}...")
model = YOLO(MODEL_ONNX, task='detect')

cap = cv2.VideoCapture(VIDEO_INPUT)

# --- 2. SETUP VIDEO SAVER ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
if fps_input == 0:
    fps_input = 30 # Safety fallback

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_demo_output3.mp4', fourcc, fps_input, (width, height))

# --- 3. SETUP CSV LOGGER ---
with open(LOG_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Class", "Confidence", "BoundingBox_Area", "Snapshot_File"])

print("🚀 SYSTEM ACTIVE! Processing video file...")

prev_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n✅ End of video reached!")
            break
            
        frame_height, frame_width = frame.shape[:2]
        total_screen_area = frame_height * frame_width

        results = model.predict(frame, conf=0.25, imgsz=320, device='cpu', verbose=False)
        r = results[0]
        
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            box_area = (x2 - x1) * (y2 - y1)
            
            color = (0, 0, 255) if class_name == "Pothole" else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if box_area > (total_screen_area * 0.02): 
                timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19] 
                
                snapshot_name = f"{class_name}_{timestamp_file}.jpg"
                snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
                
                crop_img = frame[max(0, y1):min(frame_height, y2), max(0, x1):min(frame_width, x2)]
                if crop_img.size > 0: 
                    cv2.imwrite(snapshot_path, crop_img)
                
                with open(LOG_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp_display, class_name, f"{conf:.2f}", box_area, snapshot_name])
                
                cv2.putText(frame, "SNAPPED & LOGGED!", (x2-120, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        curr_time = time.time()
        time_diff = curr_time - prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        prev_time = curr_time
        
        cv2.putText(frame, f"Pi 4 FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        cv2.imshow("Processing Video...", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Program interrupted by user (Ctrl+C).")

finally:
    print("\n💾 Safely closing and saving video file...")
    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    print(f"✅ DONE! Video saved as 'final_demo_output.mp4'")
