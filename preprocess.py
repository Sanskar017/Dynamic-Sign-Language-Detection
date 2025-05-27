import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp

# Paths
input_dir = r"D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\frames"
output_dir = r"D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\preprocessed_frames"
os.makedirs(output_dir, exist_ok=True)

# MediaPipe Selfie Segmentation setup
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Image size and augmentation setup
target_size = (224, 224)
augmentations_per_image = 3

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False  # Flip only if signs are symmetric
)

def process_and_save(img, save_path_base):
    # Background segmentation
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = segmentor.process(img_rgb)
    mask = results.segmentation_mask
    condition = mask > 0.1
    bg_image = cv2.GaussianBlur(img, (55, 55), 0)
    img_no_bg = np.where(condition[..., None], img, bg_image)

    # Resize
    resized = cv2.resize(img_no_bg, target_size)

    # Save preview image (for inspection)
    preview_path = f"{save_path_base}_preview.jpg"
    cv2.imwrite(preview_path, resized)

    # Normalize for MobileNetV2
    resized_float = resized.astype(np.float32)
    normalized = preprocess_input(resized_float)

    # Augment and save
    img_expanded = np.expand_dims(normalized, axis=0)
    aug_iter = datagen.flow(img_expanded, batch_size=1)

    for i in range(augmentations_per_image):
        aug_img = next(aug_iter)[0]

        # Convert [-1, 1] back to [0, 255] for saving
        aug_img_uint8 = ((aug_img + 1.0) * 127.5).astype(np.uint8)

        aug_path = f"{save_path_base}_aug{i+1}.jpg"
        cv2.imwrite(aug_path, aug_img_uint8)

# Main processing loop
for class_folder in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_dir, class_folder)
    os.makedirs(output_class_path, exist_ok=True)

    for file in os.listdir(class_path):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        save_path_base = os.path.join(output_class_path, os.path.splitext(file)[0])
        process_and_save(img, save_path_base)

print("✅ Preprocessing and augmentation completed successfully!")
