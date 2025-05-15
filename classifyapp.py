import os
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Paths
MODEL_PATH = '/home/carl/BarongClassification/article1model.pt'
CLASSIFIED_DIR = '/home/carl/BarongClassification/classified_images'
TEMP_IMAGE_PATH = '/home/carl/captured_image.jpg'

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Initialize Picamera2
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    return picam2

# Resize image to standard input size
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Capture image and ask user whether to keep or discard
def capture_and_ask_user(picam2):
    while True:
        image = picam2.capture_array()
        print("Image captured.")
        image_resized = resize_image(image, width=640, height=640)
        image_bgr = image_resized[:, :, ::-1]  # Convert RGB to BGR

        window_name = "Captured Image - Press 'y' to keep, 'n' to discard, 'esc' to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image_bgr)
        cv2.moveWindow(window_name, 0, 0)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("Exiting capture loop.")
            cv2.destroyAllWindows()
            return None
        elif key == ord('y'):
            print("Image accepted for classification.")
            cv2.imwrite(TEMP_IMAGE_PATH, image_bgr)
            cv2.destroyWindow(window_name)
            return TEMP_IMAGE_PATH
        elif key == ord('n'):
            print("Image discarded.")
            cv2.destroyWindow(window_name)
            continue
        else:
            print("Invalid key. Use 'y', 'n', or 'esc'.")

# Rename and store the image in classified_images/
def rename_and_save_image(image_path):
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)
    count = len([f for f in os.listdir(CLASSIFIED_DIR) if f.startswith('classify_')])
    new_path = os.path.join(CLASSIFIED_DIR, f'classify_{count + 1}.jpg')
    os.rename(image_path, new_path)
    print(f"Image saved as {new_path}")
    return new_path

# Predict class and confidence using YOLOv8
def predict(image_path):
    results = model.predict(image_path)
    class_id = results[0].probs.top1
    confidence = results[0].probs.top1conf
    return class_names[class_id], confidence

# Show classification result and wait for 'c' or 'q'
def show_classification_result(image_path, label, confidence):
    image = cv2.imread(image_path)
    cv2.putText(image, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    result_window = "Classification Result"
    cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(result_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(result_window, image)
    cv2.moveWindow(result_window, 0, 0)

    print("Press 'c' to capture another image or 'q' to quit.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quitting program.")
            cv2.destroyAllWindows()
            exit()
        elif key == ord('c'):
            print("Continuing to capture...")
            cv2.destroyWindow(result_window)
            break
        else:
            print("Invalid key. Press 'c' or 'q'.")

# Main
if __name__ == '__main__':
    print("Starting Barong Tagalog Design Classification...")
    picam2 = init_camera()

    while True:
        image_path = capture_and_ask_user(picam2)
        if image_path is None:
            print("No image selected. Exiting...")
            break

        classified_path = rename_and_save_image(image_path)
        label, confidence = predict(classified_path)
        print(f"Predicted: {label}, Confidence: {confidence:.2f}")
        show_classification_result(classified_path, label, confidence)
