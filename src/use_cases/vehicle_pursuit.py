"""
Vehicle Pursuit Module

Goal: Track a specific fleeing vehicle or suspect.
Implementation: Uses ByteTrack to assign persistent IDs to moving vehicles.
"""

import cv2
from ultralytics import YOLO

def vehicle_pursuit(source='0', weights='yolov8n.pt'):
    print("Initializing Vehicle Pursuit Module...")
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    # VisDrone Classes: 3 (car), 4 (van), 5 (truck), 9 (motor)
    VEHICLE_CLASSES = [3, 4, 5, 9]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking, filtering for only vehicle classes
        results = model.track(
            frame, 
            classes=VEHICLE_CLASSES, 
            persist=True, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            print(f"Tracking vehicles with IDs: {track_ids}")
            # In a real drone, command center selects an ID, and drone focuses on it.

        # Visualize
        annotated_frame = results[0].plot()
        cv2.imshow("Vehicle Pursuit - Tracking", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_pursuit()
