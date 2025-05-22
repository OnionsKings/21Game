import os
import shutil
import random
from pathlib import Path

def create_directory_structure(base_path):
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    return train_path, val_path

def split_dataset(source_dir, train_dir, val_dir, train_ratio=0.8):
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_dir in class_dirs:
        os.makedirs(os.path.join(train_dir, class_dir), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_dir), exist_ok=True)
        
        class_path = os.path.join(source_dir, class_dir)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_dir, class_dir, img)
            shutil.copy2(src, dst)
            
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_dir, class_dir, img)
            shutil.copy2(src, dst)
            
        print(f"Class {class_dir}:")
        print(f"  - Training images: {len(train_images)}")
        print(f"  - Validation images: {len(val_images)}")

def main():
    random.seed(42)
    source_dir = "/Users/zhengjeppesen/Desktop/project/21Game/dataset"
    new_dataset_dir = "/Users/zhengjeppesen/Desktop/project/21Game/dataset_split"
    os.makedirs(new_dataset_dir, exist_ok=True)
    train_dir, val_dir = create_directory_structure(new_dataset_dir)
    split_dataset(source_dir, train_dir, val_dir)
    print("\nDataset splitting completed!")
    print(f"Training set location: {train_dir}")
    print(f"Validation set location: {val_dir}")

if __name__ == "__main__":
    main()