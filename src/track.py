"""
Inference and Tracking script for the Drone.
This simulates what will run on the drone's edge computer.
"""

import cv2
from ultralytics import YOLO
import argparse

def run_tracking(source, weights='yolov8n.pt', show_video=True):
    """
    Run object detection and tracking on a video feed.
    
    Args:
        source (str or int): Path to video file, or camera index (e.g., 0 for webcam)
        weights (str): Path to trained model weights
        show_video (bool): Whether to display the video feed (useful for debugging, 
                           turn off for actual drone deployment)
    """
    print(f"Loading model: {weights}")
    model = YOLO(weights)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference with object tracking
        # persist=True enables tracking between frames (e.g., assigning IDs)
        # tracker='bytetrack.yaml' uses the ByteTrack algorithm (good for people/vehicles)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        if show_video:
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            cv2.imshow("Drone CV - Detection & Tracking", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and tracking for drone feed")
    parser.add_argument('--source', type=str, default='0', help='Video source (file path or camera index)')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to weights file')
    parser.add_argument('--headless', action='store_true', help='Run without showing video window')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a camera index (e.g., '0' -> 0)
    video_source = int(args.source) if args.source.isdigit() else args.source
    
    run_tracking(source=video_source, weights=args.weights, show_video=not args.headless)
