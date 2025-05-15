import cv2
import os
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = '/home/carl/BarongClassification/article1model.pt'
model = YOLO(MODEL_PATH)
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

def init_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    return picam2

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def capture_and_ask_user(picam2):
    while True:
        image = picam2.capture_array()
        print("Image captured.")
        image_resized = resize_image(image, width=640, height=640)
        image_bgr = image_resized[:, :, ::-1]

        window_name = "Captured Image - Press 'y' to keep, 'n' to discard, 'esc' to exit"
        cv2.imshow(window_name, image_bgr)
        cv2.moveWindow(window_name, 100, 100)  # Ensure the window is visible

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # Esc key
            print("Exiting capture loop.")
            return None
        elif key == ord('y'):
            print("Image accepted.")
            image_path = "/home/carl/captured_image.jpg"
            cv2.imwrite(image_path, image_bgr)
            return image_path
        elif key == ord('n'):
            print("Image discarded.")
            continue  # Re-capture another image
        else:
            print("Invalid key. Press 'y', 'n', or 'esc'.")

def rename_and_save_image(image_path):
    folder_path = '/home/carl/BarongClassification/classified_images/'
    os.makedirs(folder_path, exist_ok=True)

    count = len([f for f in os.listdir(folder_path) if f.startswith('classify_')])
    new_image_path = os.path.join(folder_path, f'classify_{count + 1}.jpg')
    os.rename(image_path, new_image_path)
    print(f"Saved as {new_image_path}")
    return new_image_path

def predict(image_path):
    results = model.predict(image_path)
    class_id = results[0].probs.top1
    confidence = results[0].probs.top1conf
    return class_names[class_id], confidence

if __name__ == '__main__':
    print("Starting Barong Design Classifier...")
    picam2 = init_camera()

    while True:
        image_path = capture_and_ask_user(picam2)
        if image_path is None:
            print("No image accepted. Try again or press ESC to quit.")
            break

        classified_image_path = rename_and_save_image(image_path)

        print("Classifying...")
        predicted_label, confidence = predict(classified_image_path)
        print(f"Label: {predicted_label}, Confidence: {confidence:.2f}")

        result_image = cv2.imread(classified_image_path)
        cv2.putText(result_image, f"Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(result_image, f"Conf: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        window_result = "Classification Result"
        cv2.imshow(window_result, result_image)
        cv2.moveWindow(window_result, 100, 200)

        print("Press 'c' to capture again, 'q' to quit.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("Quitting.")
                cv2.destroyAllWindows()
                exit()
            elif key == ord('c'):
                print("Continuing to capture...")
                break
            else:
                print("Invalid key. Use 'c' or 'q'.")
