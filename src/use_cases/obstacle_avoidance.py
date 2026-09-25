"""
Obstacle Avoidance Module (Time-to-Collision via Bounding Box Expansion)

Goal: Prevent crashes by detecting when an object is rapidly approaching the drone's flight path.
Implementation: Tracks objects across frames. If an object's bounding box area 
                increases rapidly, it calculates a low Time-To-Collision (TTC) 
                and triggers an evasive maneuver warning.
"""
import cv2
from ultralytics import YOLO

def obstacle_avoidance(source='0', weights='yolov8n.pt'):
    print("Initializing Obstacle Avoidance Module...")
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    # Dictionary to store previous bounding box areas by object ID
    # Format: {track_id: area_in_pixels}
    previous_areas = {}
    
    # Threshold for area growth (e.g., 1.15 = 15% increase in size between frames)
    # A rapid increase means the drone is flying directly toward the object.
    GROWTH_THRESHOLD = 1.15 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run multi-object tracking (ByteTrack)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        boxes = results[0].boxes
        warning_triggered = False

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                # Calculate current bounding box area
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                current_area = (x2 - x1) * (y2 - y1)
                
                if track_id in previous_areas:
                    prev_area = previous_areas[track_id]
                    
                    # If area grew significantly, the object is approaching rapidly
                    if prev_area > 0 and (current_area / prev_area) > GROWTH_THRESHOLD:
                        warning_triggered = True
                        cv2.putText(frame, f"COLLISION: ID {track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Draw a thick red warning box around the obstacle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # Update the history for the next frame comparison
                previous_areas[track_id] = current_area

        if warning_triggered:
            print("[CRITICAL] Imminent Collision Detected! Triggering evasive maneuver.")
            cv2.putText(frame, "EVASIVE MANEUVER REQUIRED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Plot standard detections alongside our custom collision graphics
        annotated_frame = results[0].plot(img=frame)
        cv2.imshow("Obstacle Avoidance - TTC Analysis", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    obstacle_avoidance()
