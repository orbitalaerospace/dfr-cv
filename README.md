# First Responder Drone - Computer Vision (dfr-cv)

This repository contains the computer vision module for a first responder drone, optimized for edge-hardware deployment.

## Dataset Information: VisDrone
The models in this repository are fine-tuned using the **VisDrone Dataset**. VisDrone is specifically captured from drone-mounted cameras at various altitudes, angles, and lighting conditions, making it perfect for aerial computer vision. Unlike standard datasets taken from eye-level, this dataset prevents the AI from failing when looking top-down.

### Classes (10 categories)
1. `pedestrian` (people walking/standing)
2. `people` (dense crowds)
3. `bicycle`
4. `car`
5. `van`
6. `truck`
7. `tricycle`
8. `awning-tricycle`
9. `bus`
10. `motor` (motorcycles/scooters)

## Use Cases
We have separated the drone's capabilities into distinct modules found in `src/use_cases/`:

1. **Search and Rescue (SAR):** (`search_and_rescue.py`) 
   Hyper-focuses on detecting pedestrians/people from high altitudes to locate missing persons in disaster zones.
2. **Vehicle Pursuit:** (`vehicle_pursuit.py`) 
   Uses Multi-Object Tracking (ByteTrack) to lock onto and track fleeing vehicles (cars, vans, motors) across frames.
3. **Crowd & Traffic Density:** (`crowd_density.py`) 
   Analyzes frame population to estimate crowd sizes and identify traffic bottlenecks.
4. **Target Following / Gimbal Control:** (`target_following.py`) 
   Calculates the relative offset of a target from the camera center to issue pitch/yaw commands to the drone's gimbal.
5. **Wildfire & Smoke Detection:** (`fire_smoke_detection.py`) 
   Provides early warning for natural disasters by identifying smoke plumes and active fire lines.
6. **Automated License Plate Recognition (ALPR):** (`alpr.py`)
   Crops detected license plates and utilizes EasyOCR to read the plate text from the air.
7. **Ground Gesture Recognition:** (`gesture_recognition.py`)
   Uses YOLO-Pose to interpret the body language of first responders on the ground to issue silent flight commands.
8. **Obstacle Avoidance:** (`obstacle_avoidance.py`)
   Calculates Time-To-Collision (TTC) by analyzing the rapid expansion of tracked bounding boxes to prevent crashes.

---

## Getting Started

### 1. Installation
```bash
# Create and activate a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download the VisDrone Dataset
Ultralytics handles the download and conversion to YOLO format automatically. Run our helper script to trigger the download:
```bash
python scripts/download_data.py
```
*(Note: This downloads ~1.5GB of image data and labels into your datasets folder).*

### 3. Running the Use Cases
You can run each module locally to test it. By default, they will connect to your computer's webcam (camera index `0`). To exit a running module, press the **`q`** key on your keyboard.

**Run Search and Rescue (SAR):**
```bash
python src/use_cases/search_and_rescue.py
```

**Run Vehicle Pursuit:**
```bash
python src/use_cases/vehicle_pursuit.py
```

**Run Crowd & Traffic Density:**
```bash
python src/use_cases/crowd_density.py
```

**Run Target Following:**
```bash
python src/use_cases/target_following.py
```

**Run Wildfire & Smoke Detection:**
```bash
python src/use_cases/fire_smoke_detection.py
```

**Run ALPR (License Plate Recognition):**
```bash
python src/use_cases/alpr.py
```

**Run Ground Gesture Recognition:**
```bash
python src/use_cases/gesture_recognition.py
```

**Run Obstacle Avoidance:**
```bash
python src/use_cases/obstacle_avoidance.py
```

### 4. Training & Fine-Tuning
If you want to train or fine-tune the YOLO model from scratch using the downloaded VisDrone dataset:
```bash
python src/train.py
```

**Want to train the drone to recognize custom objects (like Fire, Smoke, or License Plates)?** 
Read our detailed **[Fine-Tuning Guide](docs/FINE_TUNING.md)** for step-by-step instructions on data formatting, custom YAML configurations, and exporting for edge hardware.
