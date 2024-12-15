import os
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

model = YOLO('C:\\Users\\Carl\\Barong Tagalog Design Classification\\article2model.pt')

test_folder = 'D:\\School Files\\THESIS\\Barong Pictures\\OriginalPNG\\test'

class_names = ['ArtDeco', 'Ethnic', 'Special' , 'Traditional']  

results_list = []

def predict_and_store(image_path, true_label):
    img = Image.open(image_path)
    results = model.predict(image_path)
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
        'Correct': is_correct
    })

for subfolder in os.listdir(test_folder):
    subfolder_path = os.path.join(test_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for image_file in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_file)
            if os.path.isfile(image_path):
                true_label = subfolder
                predict_and_store(image_path, true_label)

df = pd.DataFrame(results_list)

df['Correct'] = df['Correct'].astype(int)
accuracy_per_image = df.groupby('Image Path')['Correct'].mean()
average_accuracy_per_class = df.groupby('True Label')['Correct'].mean()

print("Image-wise Accuracy:")
print(accuracy_per_image)
print("\nAverage Accuracy per Classification:")
print(average_accuracy_per_class)

df.to_csv('prediction_summary.csv', index=False)
print("\nPrediction summary saved to 'prediction_summary_model2.csv'.")

overall_accuracy = accuracy_score(df['True Label'], df['Predicted Label'])
print(f"\nOverall Accuracy: {overall_accuracy:.2f}")

y_true = df['True Label']
y_pred = df['Predicted Label']
conf_matrix = confusion_matrix(y_true, y_pred, labels=class_names)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix For YOLOv8 Model\nOverall Accuracy: {overall_accuracy:.2f}', pad=20)
plt.savefig('confusion_matrix_yolov8_model2.png')
print("\nConfusion matrix saved to 'confusion_matrix.png'.")
plt.show()
