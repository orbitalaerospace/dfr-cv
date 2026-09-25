"""
Ground Gesture Recognition Module

Goal: Allow first responders on the ground to command the drone using body language.
Implementation: Uses YOLO-Pose to detect skeletal keypoints and calculates relative arm positions.
"""

import cv2
import math
from ultralytics import YOLO

def calculate_angle(p1, p2, p3):
    """Calculate the angle between 3 points (x, y)"""
    if p1[0] == 0 or p2[0] == 0 or p3[0] == 0: 
        return 0 # Missing keypoint
    
    angle = math.degrees(
        math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - 
        math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    )
    return angle + 360 if angle < 0 else angle

def gesture_recognition(source='0', weights='yolov8n-pose.pt'):
    print("Initializing Ground Gesture Recognition Module...")
    # Uses the YOLOv8 Pose Estimation model
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run pose estimation
        results = model(frame, verbose=False)
        
        command = "HOVER / STANDBY"
        
        # Analyze keypoints for gestures
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Get keypoints for the first detected person
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            
            # YOLO-Pose has 17 keypoints. We need at least index 10 for wrists.
            if len(keypoints) >= 11:
                # Common YOLO-Pose Indices:
                # 5: Left Shoulder, 6: Right Shoulder
                # 7: Left Elbow,    8: Right Elbow
                # 9: Left Wrist,    10: Right Wrist
                
                l_shoulder, r_shoulder = keypoints[5], keypoints[6]
                l_wrist, r_wrist = keypoints[9], keypoints[10]
                
                # Check validity (ensure point is actually detected, x > 0)
                if l_shoulder[0] > 0 and r_shoulder[0] > 0 and l_wrist[0] > 0 and r_wrist[0] > 0:
                    
                    # Heuristic 1: Crossed Arms (X-shape over chest) -> EMERGENCY LAND
                    # Note: Y-coordinates go DOWN in OpenCV (smaller Y is "higher" on screen)
                    crossed_arms = (
                        l_wrist[1] < l_shoulder[1] and r_wrist[1] < r_shoulder[1] and 
                        l_wrist[0] > r_shoulder[0] and r_wrist[0] < l_shoulder[0]
                    )
                    
                    # Heuristic 2: One hand raised significantly above shoulder -> COME CLOSER
                    hand_raised = (
                        (l_wrist[1] < l_shoulder[1] - 50) or 
                        (r_wrist[1] < r_shoulder[1] - 50)
                    )

                    if crossed_arms:
                        command = "EMERGENCY LAND"
                    elif hand_raised:
                        command = "COME CLOSER"
                
                # Display the interpreted command
                color = (0, 0, 255) if command != "HOVER / STANDBY" else (0, 255, 0)
                cv2.putText(frame, f"COMMAND: {command}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Plot the skeleton overlay on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("Gesture Recognition - Drone Control", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_recognition()
