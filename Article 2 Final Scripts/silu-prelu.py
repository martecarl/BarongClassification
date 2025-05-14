import torch
import torch.nn as nn
from ultralytics import YOLO

def replace_activation(module):
    """Recursively replace activation functions with PReLU."""
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.Mish)):
            print(f"Replacing {name}: {child.__class__.__name__} → PReLU")
            setattr(module, name, nn.PReLU(num_parameters=1))
        else:
            replace_activation(child)

def check_prelu(module):
    """Check if all activations are now PReLU."""
    for child in module.children():
        if isinstance(child, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.Mish)):
            print(f"Found non-PReLU activation: {child.__class__.__name__}")
            return False
        elif not check_prelu(child):
            return False
    return True

def modify_yolov8_activation(model_path, save_path):
    """Modify YOLOv8 model to use PReLU and save the modified model."""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print("Replacing activation functions with PReLU...")
    replace_activation(model.model)

    print("Checking if all activations are PReLU...")
    if check_prelu(model.model):
        print("✅ All activations successfully replaced with PReLU.")
    else:
        print("⚠️ Some activations were not replaced.")

    # Save the model using the model.save() method
    print(f"Saving modified model to: {save_path}")
    model.save(save_path)  # Proper model saving with YOLO's save method
    print("✅ Modified model saved successfully.")

    # Check if the model was saved correctly by loading it back
    print("Verifying saved model integrity...")
    try:
        saved_model = YOLO(save_path)  # Try loading the saved model
        print("✅ Model loaded successfully after saving.")
        print(saved_model)  # Print basic model information to verify
    except Exception as e:
        print(f"Error loading saved model: {e}")

# Example usage
if __name__ == "__main__":
    modify_yolov8_activation("yolov8s-cls.pt", "yolov8s-cls-prelu.pt")
