import os
import cv2
from picamera2 import Picamera2, Preview
from ultralytics import YOLO  # For .pt model
import numpy as np

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Directory to store classified images
classified_images_dir = "/home/carl/classified_images"
os.makedirs(classified_images_dir, exist_ok=True)

# Initialize Picamera2
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())  # Set up for still image capture
    picam2.start()  # Start the camera
    return picam2

# Function to capture image and handle user decision
def capture_and_ask_user(picam2):
    # Capture an image
    image = picam2.capture_array()  # Capture the image as a NumPy array
    print("Image captured.")

    # Convert the image to BGR format (Picamera2 captures in RGB)
    image_bgr = image[:, :, ::-1]

    # Display the captured image for the user to decide
    cv2.imshow("Captured Image - Press 'y' to keep, 'n' to discard", image_bgr)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # Esc key to exit
        print("Exiting without capturing.")
        return None
    elif key == ord('y'):  # 'y' to keep the image
        print("Image kept for classification.")
        # Save the image
        image_path = "/home/carl/captured_image.jpg"
        cv2.imwrite(image_path, image_bgr)
        return image_path
    elif key == ord('n'):  # 'n' to discard the image
        print("Image discarded. Capturing a new image.")
        return None

# Function to rename the image and prepare for classification
def rename_and_save_image(image_path):
    # Find the next available filename
    existing_files = os.listdir(classified_images_dir)
    existing_files = [f for f in existing_files if f.startswith("classify_")]
    existing_files.sort()
    
    next_index = len(existing_files) + 1
    new_image_name = f"classify_{next_index}.jpg"
    new_image_path = os.path.join(classified_images_dir, new_image_name)
    
    os.rename(image_path, new_image_path)
    print(f"Image renamed and saved as {new_image_name}")
    return new_image_path

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
    picam2 = init_camera()  # Initialize Picamera2
    while True:
        image_path = capture_and_ask_user(picam2)
        if image_path is None:
            print("No image captured. Exiting...")
            break

        # Rename the image if it is chosen for classification
        classified_image_path = rename_and_save_image(image_path)

        print("Classifying image...")
        predicted_label, confidence = predict(classified_image_path)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")

        print("Press 'c' to capture another image or any other key to quit.")
        if cv2.waitKey(0) & 0xFF != ord('c'):
            break

    print("Exiting program. Goodbye!")
