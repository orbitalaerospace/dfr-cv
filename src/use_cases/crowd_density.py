"""
Crowd & Traffic Density Module

Goal: Estimate crowd sizes or identify traffic bottlenecks for disaster response.
Implementation: Counts total detections per frame and calculates density.
"""

import cv2
from ultralytics import YOLO

def crowd_density(source='0', weights='yolov8n.pt', crowd_threshold=15):
    print("Initializing Crowd & Traffic Density Module...")
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        
        # Count humans (0, 1) and vehicles (3, 4, 5, 9)
        boxes = results[0].boxes
        human_count = sum(1 for c in boxes.cls if c in [0, 1])
        vehicle_count = sum(1 for c in boxes.cls if c in [3, 4, 5, 9])
        total_objects = len(boxes)

        # Output metrics
        if total_objects > crowd_threshold:
            print(f"[HIGH DENSITY WARNING] {human_count} humans, {vehicle_count} vehicles detected!")
        
        # Overlay counts on frame
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Humans: {human_count} | Vehicles: {vehicle_count}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
        cv2.imshow("Crowd & Traffic Density", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    crowd_density()
