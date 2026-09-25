"""
Script to download and prepare the VisDrone dataset.
You can run this later when you are ready to get the data.

Ultralytics (YOLOv8) has a built-in mechanism to download the VisDrone dataset
and convert it to the YOLO format automatically. 
"""

import os
from ultralytics import YOLO

def download_and_prepare():
    print("Initializing YOLO model to trigger VisDrone download...")
    
    # We load a small model (nano) just to trigger the dataset download pipeline
    model = YOLO('yolov8n.pt')
    
    print("\n--- Starting Dataset Download ---")
    print("This will download approx ~1.5GB of data and extract it.")
    print("It might take a while depending on your internet connection.")
    
    # Running a 1-epoch training run on VisDrone.yaml triggers the automatic download
    # if the dataset doesn't exist locally. We set imgsz=160 to make it very fast 
    # just in case it actually starts the first epoch.
    try:
        model.train(data='VisDrone.yaml', epochs=1, imgsz=160, project='../datasets')
        print("\nDataset downloaded and prepared successfully!")
        print("It should be located in your 'datasets' directory or the ultralytics default datasets folder.")
    except Exception as e:
        print(f"\nError during download: {e}")
        print("You can also manually download it from: https://github.com/VisDrone/VisDrone-Dataset")

if __name__ == "__main__":
    download_and_prepare()
