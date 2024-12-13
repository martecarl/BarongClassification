import cv2
import os
from picamera2 import Picamera2
from ultralytics import YOLO  # For .pt model

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/yolov8n-cls.pt'

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Initialize camera
def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())  # Set up still image capture
    picam2.start()
    return picam2

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_width=1024, max_height=800):
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Function to capture and display the image with user confirmation
def capture_and_ask_user(picam2):
    captured_images = []  # List to store images temporarily

    while True:
        # Capture an image
        image = picam2.capture_array()  # Capture the image as a NumPy array
        print("Image captured.")

        # Resize the captured image to a manageable size and keep it consistent
        image_resized = resize_image(image)

        # Convert the image to BGR format (Picamera2 captures in RGB)
        image_bgr = image_resized[:, :, ::-1]

        # Display the captured image for the user to decide
        cv2.imshow("Captured Image - Press 'y' to keep, 'n' to discard, 'q' to quit", image_bgr)

        # Move the window to the center of the screen (optional)
        screen_width, screen_height = 1920, 1080  # Adjust to your screen size
        window_x = (screen_width - image_bgr.shape[1]) // 2
        window_y = (screen_height - image_bgr.shape[0]) // 2
        cv2.moveWindow("Captured Image - Press 'y' to keep, 'n' to discard, 'q' to quit", window_x, window_y)

        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):  # Esc key or 'q' to quit and discard image
            print("Exiting without capturing.")
            cv2.destroyAllWindows()
            return None, captured_images
        elif key == ord('y'):  # 'y' to keep the image
            print("Image kept for classification.")
            captured_images.append(image_bgr)  # Keep the current image
            cv2.destroyAllWindows()
            return captured_images[-1], captured_images  # Return the last captured image
        elif key == ord('n'):  # 'n' to discard the image
            print("Image discarded. Capturing a new image.")
            cv2.destroyAllWindows()

# Rename and save the image
def rename_and_save_image(image_path):
    # Generate new name like classify_1, classify_2, etc.
    folder_path = '/home/carl/BarongClassification/classified_images/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)
    count = len([file for file in existing_files if file.startswith('classify_')])

    new_image_path = os.path.join(folder_path, f'classify_{count + 1}.jpg')
    cv2.imwrite(new_image_path, image_path)  # Save the image as classify_X.jpg
    print(f"Image renamed and saved as {new_image_path}")
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
        image, captured_images = capture_and_ask_user(picam2)
        if image is None:
            print("No image captured. Continuing to capture new image...")
            continue  # Skip classification and move on to the next image capture

        # Confirm if the user wants to save the last captured image
        print("Do you want to save the last image? Press 'y' to save, 'n' to discard.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):  # Save the image if 'y' is pressed
            # Delete all previously captured images and save the latest one
            for img in captured_images[:-1]:
                del img
            print("Deleting previous discarded images...")
            classified_image_path = rename_and_save_image(image)

            print("Classifying image...")
            predicted_label, confidence = predict(classified_image_path)
            print(f"Predicted Label: {predicted_label}")
            print(f"Confidence: {confidence:.2f}")

            # Show the classification result
            result_image = cv2.imread(classified_image_path)
            cv2.putText(result_image, f"Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Classification Result - Press 'c' to capture another image or 'q' to quit", result_image)

            # Move the result window to the center
            screen_width, screen_height = 1920, 1080  # Adjust to your screen size
            window_x = (screen_width - result_image.shape[1]) // 2
            window_y = (screen_height - result_image.shape[0]) // 2
            cv2.moveWindow("Classification Result - Press 'c' to capture another image or 'q' to quit", window_x, window_y)

            # Wait for the user to press 'c' or 'q'
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):  # If 'q' is pressed, exit the loop
                    print("Exiting program.")
                    cv2.destroyAllWindows()
                    break
                elif key == ord('c'):  # If 'c' is pressed, continue capturing images
                    print("Capturing another image...")
                    cv2.destroyAllWindows()
                    break
        elif key == ord('n'):
            print("Image discarded. Capturing a new image...")
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    print("Goodbye!")
