import cv2
import time
import os
import subprocess
from ultralytics import YOLO  # For .pt model

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Camera capture and preview using OpenCV

def capture_image_with_preview():
    image_path = "/home/carl/captured_image.jpg"
    print("Launching camera preview... Press Ctrl+C to capture the image and exit preview.")
    
    try:
        # Launch libcamera-apps preview and capture
        subprocess.run(["libcamera-jpeg", "-o", image_path, "--preview", "0,0,1280,720", "-n"], check=True)
        print(f"Image captured and saved at {image_path}")
        return image_path
    except subprocess.CalledProcessError as e:
        print(f"Error capturing image: {e}")
        return None

# Predict with the model
def predict(image_path):
    results = model.predict(image_path)
    class_id = results[0].probs.top1  # Get top-1 class ID
    confidence = results[0].probs.top1conf  # Confidence score

    predicted_label = class_names[class_id]
    return predicted_label, confidence

# Main script
if __name__ == '__main__':
    print("Starting Barong Design Classification...")
    while True:
        image_path = capture_image_with_preview()
        if image_path is None:
            print("No image captured. Exiting...")
            break

        print("Classifying image...")
        predicted_label, confidence = predict(image_path)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")

        print("Press 'c' to capture another image or any other key to quit.")
        if cv2.waitKey(0) & 0xFF != ord('c'):
            break

    print("Exiting program. Goodbye!")
