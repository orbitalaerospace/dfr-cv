"""
Search and Rescue (SAR) Module

Goal: Detect missing persons in disaster zones or remote areas.
Implementation: Filters YOLO detections to ONLY look for humans (pedestrians/people).
"""

import cv2
from ultralytics import YOLO

def search_and_rescue(source='0', weights='yolov8n.pt'):
    print("Initializing Search & Rescue Module...")
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    # VisDrone Classes: 0 (pedestrian), 1 (people)
    HUMAN_CLASSES = [0, 1]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference, filtering for only human classes
        results = model(frame, classes=HUMAN_CLASSES, verbose=False)
        
        # Check if any humans were found
        detections = results[0].boxes
        if len(detections) > 0:
            print(f"[ALERT] Found {len(detections)} potential human(s)!")
            # In a real drone, you would trigger a GPS ping or command center alert here.

        # Visualize
        annotated_frame = results[0].plot()
        cv2.imshow("Search & Rescue - Human Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    search_and_rescue()
