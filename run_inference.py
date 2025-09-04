from ultralytics import YOLO
import os
from PIL import Image
import argparse

def run_inference(weights_path, source_folder):
    # Load the trained YOLO model
    model = YOLO(weights_path)

    # Run predictions on the source folder images
    results = model.predict(source=source_folder, save=True)

    # Display or log output images path
    pred_folder = "runs/detect/predict"
    print(f"Inference completed. Predictions saved in {pred_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on images.")
    parser.add_argument('--weights', type=str, required=True, help='models/best1.pt .pt file')
    parser.add_argument('--source', type=str, required=True, help='test images')
    args = parser.parse_args()

    run_inference(args.weights, args.source)

