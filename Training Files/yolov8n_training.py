from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(
    data="C:/Users/Carl/Thesis/MLDataset",
    task='classify',
    epochs=50,                  # Number of training epochs
    imgsz=640,                  # Input image size
    batch=16,                   # Batch size
    device='cpu',               # Use GPU if available
    workers=4,                  # Number of workers
    optimizer="Adam",           # Optimizer type
    lr0=0.001,                  # Learning rate
    seed=42,                    # Random seed
    verbose=True,               # Training details
    cache=False
)

print("Training data loaded:", model.data)
print("Validation data loaded:", model.val)

