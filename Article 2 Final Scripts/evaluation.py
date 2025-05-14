import time
from sklearn.metrics import precision_score, recall_score, average_precision_score
import torch
from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def evaluate_classification_metrics(model_path, data_path):
    model = YOLO(model_path)
    model.fuse()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    y_true, y_pred, y_scores = [], [], []
    total_inference_time = 0

    for images, labels in loader:
        with torch.no_grad():
            start_time = time.time()
            preds = model(images)[0].probs
            end_time = time.time()

        total_inference_time += (end_time - start_time)

        predicted_class = preds.top1
        class_scores = preds.data.tolist()

        y_true.append(labels.item())
        y_pred.append(predicted_class)
        y_scores.append(class_scores)

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    map_score = average_precision_score(y_true, y_scores, average='macro')

    avg_inference_time = total_inference_time / len(dataset)

    return precision, recall, map_score, avg_inference_time

# Example usage
if __name__ == "__main__":
    test_data_path = "C:/Users/Carl/Barong Tagalog Design Classification/Training Files/MLDataset/test"

    yolov8_path = "article1model.pt"
    prelu_path = "prelu_trained_2.pt"

    print("Evaluating Original YOLOv8:")
    prec1, rec1, map1, inf1 = evaluate_classification_metrics(yolov8_path, test_data_path)
    print(f"Precision: {prec1:.4f}, Recall: {rec1:.4f}, mAP: {map1:.4f}, Avg Inference Time: {inf1*1000:.2f} ms")

    print("\nEvaluating PReLU YOLOv8:")
    prec2, rec2, map2, inf2 = evaluate_classification_metrics(prelu_path, test_data_path)
    print(f"Precision: {prec2:.4f}, Recall: {rec2:.4f}, mAP: {map2:.4f}, Avg Inference Time: {inf2*1000:.2f} ms")
