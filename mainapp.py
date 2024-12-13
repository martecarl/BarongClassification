import os
import torch
from torchvision import transforms
from PIL import Image
import subprocess
from datetime import datetime

# Load your trained model
model_path = "model.pth"  # Replace with your model's file path
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define the class names (replace with your Barong design classifications)
class_names = ["Art Deco", "Ethnic", "Special", "Traditional"]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size based on your model's input requirements
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Capture image using libcamera
def capture_image(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(output_folder, f"barong_{timestamp}.jpg")

    # Capture image using libcamera
    capture_command = f"libcamera-jpeg -o {image_path} --width 640 --height 480"
    subprocess.run(capture_command, shell=True, check=True)

    print(f"Image captured and saved to: {image_path}")
    return image_path

# Classify image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        class_label = class_names[predicted.item()]

    print(f"Classified as: {class_label}")
    return class_label

# Main function
def main():
    output_folder = "captured_images"
    image_path = capture_image(output_folder)
    class_label = classify_image(image_path)
    print(f"Image {os.path.basename(image_path)} classified as: {class_label}")

if __name__ == "__main__":
    main()
