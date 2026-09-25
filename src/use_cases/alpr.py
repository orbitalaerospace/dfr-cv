"""
Automated License Plate Recognition (ALPR) Module

Goal: Read license plates from tracked vehicles for law enforcement.
Implementation: Uses YOLO to detect/crop plates, and EasyOCR to read the text.
"""

import cv2
from ultralytics import YOLO
try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

def run_alpr(source='0', weights='license_plate_yolov8n.pt'):
    print("Initializing ALPR Module...")
    
    if not HAS_OCR:
        print("Error: 'easyocr' is not installed. Please install it via requirements.txt.")
        return

    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"Warning: Could not load {weights}. Falling back to default for testing.")
        model = YOLO('yolov8n.pt')

    # Initialize OCR reader (using English)
    print("Loading OCR Model... (This may take a moment)")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if deployed on Jetson/GPU
    cap = cv2.VideoCapture(source)

    # Assuming custom trained weights where class 0 is 'license_plate'
    PLATE_CLASS = [0]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, classes=PLATE_CLASS, verbose=False)
        
        boxes = results[0].boxes
        for box in boxes:
            # Get bounding box coordinates for the license plate
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Crop the license plate from the frame
            plate_crop = frame[y1:y2, x1:x2]
            
            # Run OCR on the cropped plate
            if plate_crop.size > 0:
                ocr_results = reader.readtext(plate_crop)
                
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.5: # Only trust high-confidence reads
                        print(f"[ALPR HIT] Plate: {text} (Confidence: {prob:.2f})")
                        
                        # Overlay text on the main frame
                        cv2.putText(frame, f"PLATE: {text}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        annotated_frame = results[0].plot(img=frame)
        cv2.imshow("ALPR - Drone Feed", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_alpr()
