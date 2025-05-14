import os
import time
import psutil  # Import for RAM tracking
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
model = YOLO('C:\\Users\\Carl\\Barong Tagalog Design Classification\\art2_yoloprelu.pt')

# Path to test dataset
test_folder = 'D:\\School Files\\THESIS\\Barong Pictures\\OriginalPNG\\test'

# Class names
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

results_list = []
total_inference_time = 0  
total_memory_usage = 0  
num_images = 0  

def predict_and_store(image_path, true_label):
    global total_inference_time, total_memory_usage, num_images

    img = Image.open(image_path)

    # Get RAM usage before inference
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # Convert to MB

    start_time = time.time()
    results = model.predict(image_path)
    end_time = time.time()

    # Get RAM usage after inference
    mem_after = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    memory_usage = mem_after - mem_before  # RAM used during inference

    inference_time = end_time - start_time
    total_inference_time += inference_time
    total_memory_usage += memory_usage
    num_images += 1

    probs = results[0].probs
    top1 = probs.top1
    top1conf = probs.top1conf

    class_id = top1
    confidence = top1conf
    predicted_label = class_names[class_id]
    is_correct = int(predicted_label == true_label)

    results_list.append({
        'Image Path': image_path,
        'Predicted Label': predicted_label,
        'Confidence': confidence,
        'True Label': true_label,
        'Correct': is_correct,
        'Inference Time (s)': inference_time,
        'Memory Usage (MB)': memory_usage
    })

# Iterate through test dataset
for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)
            if os.path.isfile(image_path):
                true_label = subfolder
                predict_and_store(image_path, true_label)

# Convert results to DataFrame
df = pd.DataFrame(results_list)

# Save results
df.to_csv('prediction_summary_model2.csv', index=False)
print("\nPrediction summary saved to 'prediction_summary_model2.csv'.")

# Compute overall accuracy
overall_accuracy = accuracy_score(df['True Label'], df['Predicted Label'])
print(f"\nOverall Accuracy: {overall_accuracy:.2f}")

# Compute confusion matrix
y_true = df['True Label']
y_pred = df['Predicted Label']
conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix For YOLOv8 Model\nOverall Accuracy: {overall_accuracy:.2f}', pad=20)
plt.savefig('confusion_matrix_yolov8_model2.png')
print("\nConfusion matrix saved to 'confusion_matrix_yolov8_model2.png'.")
plt.show()

# Print inference time and memory usage results
average_inference_time = total_inference_time / num_images if num_images > 0 else 0
average_memory_usage = total_memory_usage / num_images if num_images > 0 else 0

print(f"\nTotal Inference Time: {total_inference_time:.2f} seconds")
print(f"Average Inference Time per Image: {average_inference_time:.4f} seconds")
print(f"Average Memory Usage per Image: {average_memory_usage:.2f} MB")
