import torch
import torch.nn as nn
from ultralytics import YOLO

def replace_activation(module):
    """Recursively replace activation functions with PReLU in the model."""
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.SiLU, nn.Mish)):
            setattr(module, name, nn.PReLU(num_parameters=1))
        else:
            replace_activation(child)

def modify_yolov8_activation(model_path, save_path):
    """Loads a YOLOv8 classification model, replaces activations with PReLU, and saves it."""
    model = YOLO(model_path)  # Load the model
    model.model.apply(replace_activation)  # Modify activations
    model.save(save_path)  # Save the modified model
    print(f"Modified model saved at: {save_path}")

# Example usage
modify_yolov8_activation("yolov8s-cls.pt", "yolov8s-cls-prelu.pt")
