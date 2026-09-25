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

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python scripts/download_data.py`
3. Train model: `python src/train.py`
