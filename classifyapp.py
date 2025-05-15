import os
import time
import cv2
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

# Load model
model = YOLO(MODEL_PATH)
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Initialize camera
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    return picam2

# Pad image to center
def center_image_on_canvas(image, canvas_width, canvas_height):
    h, w = image.shape[:2]
    top = (canvas_height - h) // 2
    bottom = canvas_height - h - top
    left = (canvas_width - w) // 2
    right = canvas_width - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# Capture and handle 'y' or 'p' smoothly
def capture_and_ask_user(picam2):
    window_name = "Captured Image - Press 'y' to keep, 'p' to retake"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(window_name, 0, 0)

    while True:
        image = picam2.capture_array()
        image_resized = cv2.resize(image, (640, 640))[:, :, ::-1]
        image_display = center_image_on_canvas(image_resized, SCREEN_WIDTH, SCREEN_HEIGHT)
        cv2.imshow(window_name, image_display)
        cv2.waitKey(1)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                print("Image accepted.")
                cv2.imwrite(TEMP_IMAGE_PATH, image_resized)
                return TEMP_IMAGE_PATH, window_name
            elif key == ord('p'):
                print("Retaking image...")
                break  # Go back to outer loop to capture again
            else:
                print("Only 'y' or 'p' allowed.")

# Save image with new numbered filename
def rename_and_save_image(image_path):
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    count = len([f for f in os.listdir(CLASSIFIED_DIR) if f.startswith('classify_')])
    new_path = os.path.join(CLASSIFIED_DIR, f'classify_{count + 1}.jpg')
    os.rename(image_path, new_path)
    return new_path

# Run prediction
def predict(image_path):
    results = model.predict(image_path)
    class_id = results[0].probs.top1
    confidence = results[0].probs.top1conf
    return class_names[class_id], confidence

# Show classification result screen
def show_classification_result(image_path, label, confidence):
    image = cv2.imread(image_path)
    cv2.putText(image, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    image_display = center_image_on_canvas(image, SCREEN_WIDTH, SCREEN_HEIGHT)

    result_window = "Classification Result"
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(result_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(result_window, 0, 0)
    cv2.imshow(result_window, image_display)
    cv2.waitKey(1)

    print("Press 'c' to capture another image.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('c'):
            print("Continuing to next image...")
            cv2.destroyWindow(result_window)
            return
        else:
            print("Only 'c' allowed.")

# Main loop
if __name__ == '__main__':
    print("Launching Barong Classifier...")
    picam2 = init_camera()

    while True:
        image_path, capture_window = capture_and_ask_user(picam2)
        if image_path is None:
            break

        label, confidence = predict(image_path)
        classified_path = rename_and_save_image(image_path)

        cv2.destroyWindow(capture_window)  # Only destroy after classification is ready
        show_classification_result(classified_path, label, confidence)

    cv2.destroyAllWindows()
