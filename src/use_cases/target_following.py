"""
Target Following / Gimbal Control Module

Goal: Keep a selected target centered in the camera frame.
Implementation: Calculates center of target bounding box vs center of camera frame.
"""

import cv2
from ultralytics import YOLO

def target_following(source='0', weights='yolov8n.pt'):
    print("Initializing Target Following Module...")
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)
    
    # Target center of frame (assuming 640x480 resolution for example)
    FRAME_CENTER_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    FRAME_CENTER_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        # For simulation, just grab the first detected object as the "target"
        if len(boxes) > 0:
            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
            
            # Calculate target center
            target_cx = int((x1 + x2) / 2)
            target_cy = int((y1 + y2) / 2)
            
            # Calculate offset from frame center
            offset_x = target_cx - FRAME_CENTER_X
            offset_y = target_cy - FRAME_CENTER_Y
            
            # Simulate Gimbal Commands based on offset
            yaw_cmd = "RIGHT" if offset_x > 20 else "LEFT" if offset_x < -20 else "CENTER"
            pitch_cmd = "DOWN" if offset_y > 20 else "UP" if offset_y < -20 else "CENTER"
            
            print(f"Target Offset: (x:{offset_x}, y:{offset_y}) -> Gimbal Command: [YAW: {yaw_cmd}, PITCH: {pitch_cmd}]")
            
            # Draw visual guides
            cv2.circle(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), 5, (0, 255, 0), -1) # Frame center
            cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)           # Target center
            cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y), (target_cx, target_cy), (255, 0, 0), 2)
            
        annotated_frame = results[0].plot(img=frame)
        cv2.imshow("Target Following - Gimbal Simulation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    target_following()
