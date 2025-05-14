import time
import torch
import psutil
from ultralytics import YOLO

def train_model(model_path, data_path):
    """Train a YOLOv8 classification model and measure training time and memory usage."""
    model = YOLO(model_path)
    
    start_time = time.time()
    process = psutil.Process()
    
    model.train(
        data=data_path,
        task='classify',
        epochs=10,
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

    end_time = time.time()
    training_time = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    
    return training_time, memory_usage, model

def evaluate_model(model, data_path):
    """Evaluate the trained model and return accuracy metrics."""
    results = model.val(data=data_path)
    top1_acc = results.top1
    top5_acc = results.top5
    return top1_acc, top5_acc

if __name__ == "__main__":
    data_path = "C:/Users/Carl/Barong Tagalog Design Classification/Training Files/MLDataset"

    print("Training regular YOLOv8 model...")
    time_regular, memory_regular, model_regular = train_model("yolov8s-cls.pt", data_path)
    acc1_regular, acc5_regular = evaluate_model(model_regular, data_path)
    
    print("Training PReLU YOLOv8 model...")
    time_prelu, memory_prelu, model_prelu = train_model("yolov8s-cls-prelu.pt", data_path)
    acc1_prelu, acc5_prelu = evaluate_model(model_prelu, data_path)
    
    print("\nTraining Summary:")
    print(f"Regular YOLOv8 - Time: {time_regular:.2f}s, Memory: {memory_regular:.2f}MB, Top-1 Acc: {acc1_regular:.4f}, Top-5 Acc: {acc5_regular:.4f}")
    print(f"PReLU YOLOv8  - Time: {time_prelu:.2f}s, Memory: {memory_prelu:.2f}MB, Top-1 Acc: {acc1_prelu:.4f}, Top-5 Acc: {acc5_prelu:.4f}")
