import torch
import time
import os
import psutil
import gc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from ultralytics import YOLO

# Define dataset paths
train_data_path = "C:\\Users\\Carl\\Barong Tagalog Design Classification\\Training Files\\MLDataset\\train"  # Replace with the path to your dataset
val_data_path = "C:\\Users\\Carl\\Barong Tagalog Design Classification\\Training Files\\MLDataset\\val"  # Replace with the path to your validation dataset

# Define models
model_paths = {
    "Original": "yolov8s-cls.pt",  # Replace with your original model
    "Modified_PReLU": "yolov8s-cls-prelu.pt"  # Replace with your modified model
}

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust input size if needed
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
])

# Function to load datasets
def load_data(train_dir, val_dir, batch_size=32):
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Load data
train_loader, val_loader = load_data(train_data_path, val_data_path)

# Function to measure inference time
def measure_inference_time(model, iterations=50):
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(iterations):
            _ = model(torch.randn(1, 3, 224, 224))  # Example input
        end = time.time()
    avg_time = (end - start) / iterations
    return avg_time

# Function to measure memory usage
def measure_memory_usage(model):
    torch.cuda.empty_cache()
    gc.collect()
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    
    model(torch.randn(1, 3, 224, 224))  # Run one inference pass
    
    final_memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    memory_used = final_memory - initial_memory
    return memory_used

# Function to track loss convergence
def track_loss_convergence(model, train_loader, val_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses

# Main script
results = {}

for name, path in model_paths.items():
    print(f"\nEvaluating {name} Model...\n")
    model = YOLO(path)

    # Train the model directly with dataset paths (no YAML file)
    model.train(
        data={"train": train_data_path, "val": val_data_path},  # Pass data paths directly
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

    # Measure inference time
    inf_time = measure_inference_time(model)
    print(f"Inference Time: {inf_time:.6f} seconds")

    # Measure memory usage
    memory_usage = measure_memory_usage(model)
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Measure model size
    model_size = os.path.getsize(path) / (1024 ** 2)  # Convert bytes to MB
    print(f"Model Size: {model_size:.2f} MB")

    # Track loss convergence
    train_losses, val_losses = track_loss_convergence(model, train_loader, val_loader)
    results[name] = {"Inference Time": inf_time, "Memory Usage": memory_usage, "Model Size": model_size, 
                     "Train Loss": train_losses, "Val Loss": val_losses}

# Plot loss curves
plt.figure(figsize=(8, 5))
for name, data in results.items():
    plt.plot(range(1, len(data["Train Loss"]) + 1), data["Train Loss"], label=f"{name} - Train Loss")
    plt.plot(range(1, len(data["Val Loss"]) + 1), data["Val Loss"], label=f"{name} - Val Loss", linestyle="--")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence Comparison")
plt.legend()
plt.show()
