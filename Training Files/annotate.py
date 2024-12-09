import os

# Define class names and their corresponding labels
class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']

# Define the base directories for the train, val, and test sets
base_dir = "C:/Users/Carl/Thesis/MLDataset/"
directories = ['train', 'val', 'test']

# Function to create a label file for each image in a directory
def create_labels_for_images():
    for dir_type in directories:
        # Define the path for each set (train, val, test)
        dataset_dir = os.path.join(base_dir, dir_type)
        
        # Iterate through each class folder in the dataset
        for class_index, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            
            # Ensure the class directory exists
            if not os.path.exists(class_dir):
                continue
            
            # Get all image files in the class directory
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                
                # Only process image files (you can add more image extensions if needed)
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label_file_name = img_file.split('.')[0] + '.txt'
                    label_file_path = os.path.join(class_dir, label_file_name)
                    
                    # Write the class label (class_index) to the .txt file
                    with open(label_file_path, 'w') as label_file:
                        label_file.write(str(class_index))
                        print(f"Created label file for {img_file} with label {class_index} in {class_name}.")

# Run the function to create label files
create_labels_for_images()
