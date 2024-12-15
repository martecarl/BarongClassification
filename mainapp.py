import cv2
from picamera2 import Picamera2
from ultralytics import YOLO  # For .pt model

# Path to your YOLOv8 .pt model
MODEL_PATH = '/home/carl/BarongClassification/article1model.pt'

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

# Function to resize image
def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

# Function to capture image and handle user decision
def capture_and_ask_user(picam2):
    # Capture an image
    image = picam2.capture_array()  # Capture the image as a NumPy array
    print("Image captured.")

    # Resize the captured image to a manageable size
    image_resized = resize_image(image, width=640, height=640)

    # Convert the image to BGR format (Picamera2 captures in RGB)
    image_bgr = image_resized[:, :, ::-1]

    # Display the captured image for the user to decide
    cv2.imshow("Captured Image - Press 'y' to keep, 'n' to discard", image_bgr)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # Esc key to exit
        print("Exiting without capturing.")
        cv2.destroyAllWindows()
        return None
    elif key == ord('y'):  # 'y' to keep the image
        print("Image kept for classification.")
        # Save the image
        image_path = "/home/carl/captured_image.jpg"
        cv2.imwrite(image_path, image_bgr)
        cv2.destroyAllWindows()
        return image_path
    elif key == ord('n'):  # 'n' to discard the image
        print("Image discarded. Capturing a new image.")
        cv2.destroyAllWindows()
        return None

# Rename and save the image
def rename_and_save_image(image_path):
    # Generate new name like classify_1, classify_2, etc.
    import os
    folder_path = '/home/carl/BarongClassification/classified_images/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    existing_files = os.listdir(folder_path)
    count = len([file for file in existing_files if file.startswith('classify_')])

    new_image_path = os.path.join(folder_path, f'classify_{count + 1}.jpg')
    os.rename(image_path, new_image_path)
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
        image_path = capture_and_ask_user(picam2)
        if image_path is None:
            print("No image captured. Continuing to capture new image...")
            continue  # Skip classification and move on to the next image capture

        # Rename the image if it is chosen for classification
        classified_image_path = rename_and_save_image(image_path)

        print("Classifying image...")
        predicted_label, confidence = predict(classified_image_path)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.2f}")

        # Show the classification result
        result_image = cv2.imread(classified_image_path)
        cv2.putText(result_image, f"Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(result_image, f"Confidence: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Classification Result - Press 'c' to capture another image or 'q' to quit", result_image)

        # Wait for the user to press 'c' or 'q'
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # If 'q' is pressed, exit the loop
            print("Exiting program.")
            break
        elif key == ord('c'):  # If 'c' is pressed, continue capturing images
            print("Capturing another image...")
            cv2.destroyAllWindows()
            continue

    cv2.destroyAllWindows()
    print("Goodbye!")
