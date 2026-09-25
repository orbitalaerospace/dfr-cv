"""
Wildfire & Smoke Detection Module

Goal: Provide early warning for wildfires by detecting smoke plumes or active fires.
Implementation: Uses a YOLO model trained on fire/smoke datasets.
"""

import cv2
from ultralytics import YOLO

def fire_smoke_detection(source='0', weights='fire_smoke_yolov8n.pt'):
    print("Initializing Wildfire & Smoke Detection Module...")
    print(f"Note: This requires a model trained on fire/smoke data (e.g., {weights})")
    
    # We use a try-except block here so the script doesn't crash if the user 
    # hasn't downloaded the specific fire/smoke weights yet.
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"Warning: Could not load {weights}. Falling back to default for testing.")
        model = YOLO('yolov8n.pt')
        
    cap = cv2.VideoCapture(source)

    # Assuming standard custom dataset where 0='fire' and 1='smoke'
    TARGET_CLASSES = [0, 1]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, classes=TARGET_CLASSES, verbose=False)
        
        # Count detections
        boxes = results[0].boxes
        if len(boxes) > 0:
            fire_count = sum(1 for c in boxes.cls if int(c) == 0)
            smoke_count = sum(1 for c in boxes.cls if int(c) == 1)
            
            if fire_count > 0 or smoke_count > 0:
                print(f"[CRITICAL ALERT] Detected {fire_count} fire(s) and {smoke_count} smoke plume(s)!")
                # In production, this would trigger an immediate command center notification

        annotated_frame = results[0].plot()
        cv2.imshow("Wildfire & Smoke Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fire_smoke_detection()
