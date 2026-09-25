# How to Fine-Tune YOLO for Specific Drone Use Cases

If you want to train the drone to recognize *new* objects (e.g., Fire, Smoke, specific License Plates, or custom uniforms), you need to fine-tune the base YOLO model. 

Here is the step-by-step process.

## Step 1: Collect and Annotate Data
You need images (preferably from a drone's perspective/aerial view) of the objects you want to detect.
1. Collect images and place them in a folder.
2. Use a free annotation tool like [Roboflow](https://roboflow.com/) or [CVAT](https://cvat.ai/) to draw bounding boxes around your targets.
3. Export the dataset in **YOLOv8 PyTorch format**.

Your exported folder structure must look like this:
```text
custom_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Step 2: Create a Dataset Configuration (`.yaml`)
Create a YAML file (e.g., `configs/fire_smoke.yaml`) that tells YOLO where your data is and what classes exist in your labels.

```yaml
# configs/fire_smoke.yaml
path: ../datasets/custom_dataset # Absolute or relative path to your dataset root
train: images/train
val: images/val

# Number of classes
nc: 2

# Class names (must match the ID order you used during annotation)
names:
  0: fire
  1: smoke
```

## Step 3: Run the Training Script
You can use the ultralytics CLI or the Python API. For drones, **always start with the `nano` (`n`) model**, as it runs the fastest on edge hardware while preserving battery life.

**Using the Command Line:**
```bash
yolo task=detect mode=train data=configs/fire_smoke.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16
```

**Using Python:**
```python
from ultralytics import YOLO

# Load a pretrained base model (downloads automatically)
model = YOLO('yolov8n.pt') 

# Train on your custom dataset configuration
model.train(data='configs/fire_smoke.yaml', epochs=100, imgsz=640, batch=16, device='0')
```

## Step 4: Locate and Move Your New Weights
Once training finishes, Ultralytics will save your new model weights in the `runs/` directory.
Look for the file located at: `runs/detect/train/weights/best.pt`.

Copy `best.pt` into your project root and rename it to something recognizable (e.g., `fire_smoke_yolov8n.pt`).

## Step 5: Export for Edge Hardware (Crucial for Drones)
Drones usually run on companion computers like NVIDIA Jetsons or Raspberry Pis. A raw `.pt` PyTorch file is heavy. You should export your weights to a highly-optimized format like **ONNX** or **TensorRT** to maximize your Frames Per Second (FPS).

```bash
# Export to ONNX (Widely supported across Raspberry Pi, Orange Pi, Intel)
yolo export model=fire_smoke_yolov8n.pt format=onnx

# Export to TensorRT (Best for NVIDIA Jetson Nano / Orin series)
yolo export model=fire_smoke_yolov8n.pt format=engine device=0
```

## Step 6: Plug it into the Pipeline
Now you can pass your newly trained model into any of our use-case scripts!
```bash
# Example: Using the ONNX format in our fire detection script
python src/use_cases/fire_smoke_detection.py --weights fire_smoke_yolov8n.onnx
```
