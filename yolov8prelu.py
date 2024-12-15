import torch
import torch.nn as nn
from ultralytics import YOLO

# Define a custom convolutional layer with PReLU
class ConvPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              kernel_size // 2 if padding is None else padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# Function to replace activation functions with PReLU
def replace_activation_with_prelu(model):
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            groups = module.groups
            new_layer = ConvPReLU(in_channels, out_channels, kernel_size, stride, padding, groups)
            setattr(model, module_name, new_layer)

# Load and modify the YOLOv8 model
def modify_and_save_model(original_model_path, modified_model_path):
    original_model = YOLO(original_model_path)
    modified_model = original_model.model
    replace_activation_with_prelu(modified_model)
    original_model.model = modified_model
    original_model.save(modified_model_path)
    print(f"Modified model saved to {modified_model_path}")

# Modify and save the model
if __name__ == "__main__":
    modify_and_save_model('yolov8s-cls.pt', 'yolov8scls_prelu.pt')
