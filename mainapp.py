from picamera2 import Picamera2, Preview
from ultralytics import YOLO  # For .pt model
from PIL import Image
import numpy as np
import time

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Initialize the Picamera2
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())  # Set up live preview
    picam2.start_preview(Preview.QTGL)  # Use OpenGL for preview window (might require QT)
    picam2.start()  # Start the camera
    return picam2

# Capture image with live preview using Picamera2
def capture_image_with_live_preview(picam2):
    print("Live preview started. Press 'Spacebar' to capture an image or 'Esc' to exit.")
    image_path = "/home/carl/captured_image.jpg"

    while True:
        # Capture a frame
        frame = picam2.capture_array()
        
        # Display the live feed (using PIL for display)
        img = Image.fromarray(frame)
        img.show()

        # Wait for user input
        key = input("Press 'Spacebar' to capture or 'Esc' to exit: ").strip().lower()
        if key == 'esc':  # Exit without capturing
            print("Exiting without capturing.")
            return None
        elif key == ' ':  # Spacebar to capture
            # Save the captured image
            img.save(image_path)
            print(f"Image captured and saved at {image_path}")
            break

    return image_path

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
    
    # Initialize the camera
    picam2 = init_camera()

    while True:
        image_path = capture_image_with_live_preview(picam2)
        if image_path is None:
            print("No image captured. Exiting...")
            break

        print("Classifying image...")
        predicted_label, confidence = predict(image_path)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")

        # Ask user if they want to capture another image
        again = input("Press 'c' to capture another image or any other key to quit: ").strip().lower()
        if again != 'c':
            break

    print("Exiting program. Goodbye!")
