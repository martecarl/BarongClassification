import os
import shutil
from sklearn.model_selection import train_test_split

# Input directory with categorized images
input_dir = './AugDataset'
output_dir = './MLDataset'

# Categories
categories = os.listdir(input_dir)

# Output directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Ensure output directories exist
for subdir in [train_dir, val_dir, test_dir]:
    for category in categories:
        os.makedirs(os.path.join(subdir, category), exist_ok=True)

# Split dataset
for category in categories:
    category_dir = os.path.join(input_dir, category)
    images = [os.path.join(category_dir, img) for img in os.listdir(category_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Split the dataset
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.333, random_state=42)  # 20% val, 10% test

    # Move files to respective folders
    for phase, files in zip(['train', 'val', 'test'], [train, val, test]):
        for file_path in files:
            shutil.copy(file_path, os.path.join(output_dir, phase, category))

print("Dataset splitting complete!")
