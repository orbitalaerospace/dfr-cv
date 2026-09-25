"""
Training script for fine-tuning YOLO on the VisDrone dataset.
"""

from ultralytics import YOLO
import argparse

def train_model(epochs=50, batch_size=16, model_size='n'):
    """
    Train a YOLO model on the VisDrone dataset.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        model_size (str): Size of the YOLOv8 model ('n', 's', 'm', 'l', 'x').
                          'n' (nano) is recommended for edge/drone deployment.
    """
    model_name = f'yolov8{model_size}.pt'
    print(f"Loading base model: {model_name}")
    
    # Load a pretrained YOLO model
    model = YOLO(model_name)

    print("Starting training on VisDrone dataset...")
    # Train the model. VisDrone.yaml is built into ultralytics.
    results = model.train(
        data='VisDrone.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=640,          # 640 is standard, but you can increase it for small object detection
        device='0',         # Set to 'cpu' if you don't have a GPU on your training machine
        project='runs/train',
        name='dfr_visdrone_model',
        optimizer='auto',
        verbose=True
    )
    
    print("\nTraining complete! Model saved to runs/train/dfr_visdrone_model/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train First Responder Drone CV Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch, model_size=args.size)
