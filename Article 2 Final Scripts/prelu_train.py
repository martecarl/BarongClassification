import time
import psutil
from ultralytics import YOLO
import os

def train_model(model_path, data_path, save_name="trained_prelu_model.pt", log_file="training_log.txt"):
    """Train a YOLOv8 classification model, save the trained model, and log training details."""
    model = YOLO(model_path)
    start_time = time.time()
    process = psutil.Process()

    # Create / overwrite the training log file
    with open(log_file, "w") as log:
        # Start training
        results = model.train(
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

        # Log results if available
        log.write("\nTraining Metrics Summary:\n")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                log.write(f"{key}: {value}\n")

    end_time = time.time()
    training_time = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 ** 2)  # MB

    # Save trained model
    model.save(save_name)
    print(f"Trained model saved as: {save_name}")
    print(f"Training log saved to: {log_file}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")

    return training_time, memory_usage, model


if __name__ == "__main__":
    data_path = "C:/Users/Carl/Barong Tagalog Design Classification/Training Files/MLDataset"
    model_path = "yolov8s-cls-prelu.pt"
    save_name = "prelu_trained_2.pt"
    log_file = "prelu_training_log_2.txt"

    print("Starting training for PReLU-modified YOLOv8 classification model...\n")
    training_time, memory_usage, trained_model = train_model(model_path, data_path, save_name, log_file)
