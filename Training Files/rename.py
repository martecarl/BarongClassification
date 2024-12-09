import os

def rename_images_in_folder(folder_path, class_name):
    # Get all the files in the folder
    files = os.listdir(folder_path)
    # Filter out files that are not images (you can add more extensions if necessary)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    # Rename each image
    for idx, image_file in enumerate(image_files, start=1):
        # Get the file extension
        file_extension = os.path.splitext(image_file)[1]
        # Create the new name
        new_name = f"{class_name}{idx}{file_extension}"
        # Construct the full path of old and new files
        old_path = os.path.join(folder_path, image_file)
        new_path = os.path.join(folder_path, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {image_file} to {new_name}")

def rename_images_in_dataset(dataset_root):
    # Define the subfolders you want to rename
    subfolders = ['train', 'val', 'test']
    class_names = ['ArtDeco', 'Ethnic', 'Special', 'Traditional']  # Replace with your actual class names

    # Loop through each subfolder and class
    for subfolder in subfolders:
        for class_name in class_names:
            folder_path = os.path.join(dataset_root, subfolder, class_name)
            if os.path.exists(folder_path):
                print(f"Renaming images in {folder_path}...")
                rename_images_in_folder(folder_path, class_name)
            else:
                print(f"Folder {folder_path} does not exist.")

# Define the root path of your dataset
dataset_root = "C:/Users/Carl/Thesis/MLDataset"  # Change this to the root folder of your dataset

# Start renaming images in the dataset
rename_images_in_dataset(dataset_root)

print("Renaming process completed.")
