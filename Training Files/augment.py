import cv2
import albumentations as A
import os

input_dirs = {
    'art': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/ArtDeco',
    'eth': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Ethnic',
    'spec': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Special',
    'trad': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Traditional'
}

output_dirs = {
    'affine': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_affine',
    'channels': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_channels',
    'color': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_color',
    'distort': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_distort',
    'geometric': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_geometric',
    'perspective': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_perspective',
    'shadows': 'D:/School Files/THESIS/Barong Pictures/OriginalPNG/Augment_shadows'
}

transforms = {
    'affine': A.Compose([A.ElasticTransform(alpha=1, sigma=50, p=0.3)]),
    'channels': A.Compose([A.ChannelShuffle(p=0.1), A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3)]),
    'color': A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3)]),
    'distort': A.Compose([A.GridDistortion(num_steps=5, distort_limit=0.03, p=0.3)]),
    'geometric': A.Compose([A.Rotate(limit=10, p=0.5), A.HorizontalFlip(p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=5, p=0.5)]),
    'perspective': A.Compose([A.Perspective(scale=(0.02, 0.05), p=0.3)]),
    'shadows': A.Compose([A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, p=0.3)])
}

# Create output directories
for directory in output_dirs.values():
    os.makedirs(directory, exist_ok=True)

# Process images
for category, input_dir in input_dirs.items():
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        for aug_name, transform in transforms.items():
            augmented = transform(image=image)
            aug_image = augmented["image"]
            output_path = os.path.join(output_dirs[aug_name], f"{aug_name}_{category}_{filename}")
            cv2.imwrite(output_path, aug_image)
