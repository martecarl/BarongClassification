import torch
from ultralytics import YOLO

# Ensure the custom ConvPReLU class is available when loading the model

class ConvPReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvPReLU, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return self.prelu(self.conv(x))

# Train the modified model
def train_model(model_path, data_path):
    # Make sure the custom ConvPReLU is available in the current namespace
    model = YOLO('yolov8scls_prelu.pt')
    model.train(
        data=data_path,
        task='classify',
        epochs=20,
        imgsz=640,
        batch=16,
        device='cpu',
        workers=4,
        optimizer="Adam",
        lr0=0.001,
        seed=42,
        verbose=True,
        cache=False
    )
    print("Training completed.")

if __name__ == "__main__":
    train_model('yolov8scls_prelu.pt', "C:\\Users\\Carl\\Barong Tagalog Design Classification\\Training Files\\MLDataset")
