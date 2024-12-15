from ultralytics import YOLO

def train_model(data_path):
    # Load the YOLOv8 classification model
    model = YOLO('yolov8s-cls.pt')

    # Train the model
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
    # Update the data_path to point to your dataset
    data_path = "C:/Users/Carl/Barong Tagalog Design Classification/Training Files/MLDataset"
    train_model(data_path)
