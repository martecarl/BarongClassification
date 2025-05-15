import os
import time
import cv2
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Paths
MODEL_PATH = '/home/carl/BarongClassification/article1model.pt'
CLASSIFIED_DIR = '/home/carl/BarongClassification/classified_images'
TEMP_IMAGE_PATH = '/home/carl/captured_image.jpg'

# Screen resolution
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Initialize camera
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    return picam2

# Pad image with black borders to center it
def center_image_on_canvas(image, canvas_width, canvas_height):
    h, w = image.shape[:2]
    top = (canvas_height - h) // 2
    bottom = canvas_height - h - top
    left = (canvas_width - w) // 2
    right = canvas_width - w - left
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# Capture image and ask user if they want to keep it
def capture_and_ask_user(picam2):
    window_name = "Captured Image - Press 'y' to keep, 'n' to quit, any key to retake"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)

    while True:
        image = picam2.capture_array()
        print("Image captured.")
        image_resized = cv2.resize(image, (640, 640))[:, :, ::-1]
        image_display = center_image_on_canvas(image_resized, SCREEN_WIDTH, SCREEN_HEIGHT)

        cv2.imshow(window_name, image_display)
        cv2.waitKey(1)  # Ensure window gets focus

        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            print("Image accepted. Saving and classifying...")
            cv2.imwrite(TEMP_IMAGE_PATH, image_resized)
            return TEMP_IMAGE_PATH
        elif key == ord('n'):
            print("User chose to quit.")
            return 'quit'
        else:
            print("Retaking image...")
            continue  # Retain window without flickering

# Rename image as classify_#.jpg
def rename_and_save_image(image_path):
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    count = len([f for f in os.listdir(CLASSIFIED_DIR) if f.startswith('classify_')])
    new_path = os.path.join(CLASSIFIED_DIR, f'classify_{count + 1}.jpg')
    os.rename(image_path, new_path)
    print(f"Image saved as {new_path}")
    return new_path

# Run YOLOv8 model and get label + confidence
def predict(image_path):
    results = model.predict(image_path)
    class_id = results[0].probs.top1
    confidence = results[0].probs.top1conf
    return class_names[class_id], confidence

# Show result fullscreen and wait for user to press 'c'
def show_classification_result(image_path, label, confidence):
    image = cv2.imread(image_path)
    cv2.putText(image, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    image_display = center_image_on_canvas(image, SCREEN_WIDTH, SCREEN_HEIGHT)

    result_window = "Classification Result"
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(result_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(result_window, image_display)
    cv2.moveWindow(result_window, 0, 0)
    cv2.waitKey(1)  # Force window focus

    print("Press 'c' to capture another image.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            print("Continuing to capture...")
            cv2.destroyWindow(result_window)
            return
        else:
            print("Invalid key. Press 'c' to continue.")

# Main loop
if __name__ == '__main__':
    print("Starting Barong Tagalog Design Classification...")
    picam2 = init_camera()

    capture_window = None
    while True:
        image_path = capture_and_ask_user(picam2)
        if image_path == 'quit':
            print("Exiting program.")
            break

        if image_path is None:
            print("No image selected. Exiting...")
            break

        classified_path = rename_and_save_image(image_path)
        label, confidence = predict(classified_path)
        print(f"Predicted: {label}, Confidence: {confidence:.2f}")

        show_classification_result(classified_path, label, confidence)

        # Let OpenCV settle before re-looping
        cv2.waitKey(1)

    cv2.destroyAllWindows()
