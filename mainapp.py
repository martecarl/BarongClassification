import cv2
import time
from ultralytics import YOLO  # For .pt model

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Camera capture and preview
def capture_image_with_preview():
    cap = cv2.VideoCapture(0)  # Open the camera (adjust index if needed)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return None

    print("Press 's' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Show the live preview
        cv2.imshow('Camera Preview', frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' to save the image
            timestamp = int(time.time())
            image_path = f'captured_image_{timestamp}.jpg'
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            cap.release()
            cv2.destroyAllWindows()
            return image_path
        elif key == ord('q'):  # 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
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

        # Optionally delete the captured image after classification
        # os.remove(image_path)

        print("Press 'c' to capture another image or any other key to quit.")
        if cv2.waitKey(0) & 0xFF != ord('c'):
            break

    print("Exiting program. Goodbye!")
