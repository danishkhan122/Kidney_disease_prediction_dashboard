import torch
from roboflow import Roboflow
from ultralytics import YOLO
import os
import shutil


def train_kidne_model():
    # Initialize Roboflow with your API key
    rf = Roboflow(api_key="Ua713EzAz1BigIXyqpoi")

    # Access the 'kidne' project from 'phase-1' workspace
    project = rf.workspace("phase-1").project("kidne")
    version = project.version(2)

    # Download dataset in YOLOv8 format
    dataset = version.download("yolov8")
    data_yaml_path = os.path.join(dataset.location, "data.yaml")

    # Detect and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Training on device: {device}")

    # Load pretrained YOLOv8 small model
    model = YOLO("yolov8s.pt")

    # Train the model
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        device=device,
        batch=16,
        name="kidne_model"
    )

    # Save the best model weights
    save_dir = "C:/Users/Danish/Desktop/kidne-trained-model"
    os.makedirs(save_dir, exist_ok=True)
    best_weights_path = "runs/train/kidne_model/weights/best.pt"

    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, os.path.join(save_dir, "best.pt"))
        print(f"✅ Best model saved to: {save_dir}")
    else:
        print("⚠️ Training finished but best model not found!")


if __name__ == "__main__":
    train_kidne_model()
