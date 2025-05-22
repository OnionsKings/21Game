import cv2
import numpy as np
from albumentations import RandomBrightnessContrast, ISONoise, Blur, Sharpen, HueSaturationValue, Compose
import os
import random

# Function to increase brightness
def increase_brightness(image):
    transform = RandomBrightnessContrast(brightness_limit=(0.2, 0.5), contrast_limit=0, p=1.0)
    augmented = transform(image=image)
    return augmented["image"]

# Function to lower brightness
def lower_brightness(image):
    transform = RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), contrast_limit=0, p=1.0)
    augmented = transform(image=image)
    return augmented["image"]

# Function to add Salt and Pepper Noise
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0
    
    return noisy_image

# Function to apply color jittering
def color_jittering(image):
    transform = Compose([
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.7),
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
    ])
    augmented = transform(image=image)
    return augmented["image"]

# Function to apply blur
def apply_blur(image):
    transform = Blur(blur_limit=7, p=1.0)
    augmented = transform(image=image)
    return augmented["image"]

# Function to apply sharpening
def apply_sharpening(image):
    transform = Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)
    augmented = transform(image=image)
    return augmented["image"]

# Function to apply chosen augmentation
def apply_augmentation(image, choice):
    if choice == '1':
        return increase_brightness(image)
    elif choice == '2':
        return lower_brightness(image)
    elif choice == '3':
        return add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1)
    elif choice == '4':
        return color_jittering(image)
    elif choice == '5':
        return apply_blur(image)
    elif choice == '6':
        return apply_sharpening(image)
    else:
        print("Invalid choice.")
        return image

# Function to process images in bulk
def process_images_bulk(root_dir):
    augmentation_choices = ['1', '2', '3', '4', '5', '6']

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(dirpath, filename)
                image = cv2.imread(input_image_path)
                if image is None:
                    print(f"Warning: Unable to load image at {input_image_path}. Skipping this file.")
                    continue
                choice = random.choice(augmentation_choices)
                augmented_image = apply_augmentation(image, choice)
                name, ext = os.path.splitext(filename)
                output_image_path = os.path.join(dirpath, f"{name}_aug{ext}")
                cv2.imwrite(output_image_path, augmented_image)
                print(f"Augmented image saved as {output_image_path}")

if __name__ == "__main__":
    # Usage
    input_dir = "/Users/zhengjeppesen/Desktop/project/21Game/dataset"
    process_images_bulk(input_dir)
