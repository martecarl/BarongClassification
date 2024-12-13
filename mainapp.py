import cv2
from ultralytics import YOLO  # For .pt model

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)  # Load the YOLOv8 model

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Camera capture and live preview using OpenCV
def capture_image_with_live_preview():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    print("Live preview started. Press 'Spacebar' to capture an image or 'Esc' to exit.")
    image_path = "/home/carl/captured_image.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from the camera.")
            break

        # Display the live feed
        cv2.imshow("Live Preview - Press Spacebar to Capture", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            print("Exiting without capturing.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # Spacebar to capture
            # Save the captured image
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved at {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
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
    while True:
        image_path = capture_image_with_live_preview()
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
