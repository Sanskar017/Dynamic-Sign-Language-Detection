import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, RandomFlip, RandomZoom
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import json
import matplotlib.pyplot as plt

# Load the data
x2 = np.load(r"D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\x2_filtered.npy")
y2 = np.load(r"D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\y2_filtered.npy")
classes1 = np.load(r"D:\archive (3)\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Preprocessed data\classes1.npy")

# ===== CRITICAL FIXES =====
# 1. Verify and adjust class labels
print("Original class labels range:", np.min(y2), "to", np.max(y2))
print("Number of classes in classes1:", len(classes1))

# Ensure labels are zero-based and contiguous
unique_labels = np.unique(y2)
print("Unique labels in y2:", unique_labels)
print("Max label value:", np.max(y2))

# If labels don't start from 0 or have gaps, remap them
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y2 = le.fit_transform(y2)  # This will make labels 0, 1, 2, ..., n_classes-1

# Update parameters
IMG_SIZE = x2.shape[2]
SEQUENCE_LENGTH = x2.shape[1]
num_classes = len(np.unique(y2))  # Use actual number of unique classes
print("Adjusted number of classes:", num_classes)

# ===== DATA SPLITTING =====
# Create video IDs (adjust based on your actual data)
video_ids = np.arange(len(x2)) // 10  # Assuming 10 sequences per video

# Split data by video IDs
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(x2, y2, groups=video_ids))

X_train, X_test = x2[train_idx], x2[test_idx]
y_train, y_test = y2[train_idx], y2[test_idx]

# Verify splits
print("\nData Split Verification:")
print("Unique videos in train:", len(np.unique(video_ids[train_idx])))
print("Unique videos in test:", len(np.unique(video_ids[test_idx])))
print("Total unique videos:", len(np.unique(video_ids)))
print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

# ===== MODEL BUILDING =====
# Adjust MobileNetV2 input size if needed
if IMG_SIZE not in [96, 128, 160, 192, 224]:
    print(f"\nWarning: Resizing images from {IMG_SIZE} to 224 for MobileNetV2")
    from tensorflow.keras.layers import Resizing
    resize_layer = Resizing(224, 224)
else:
    resize_layer = tf.identity  # No resizing needed

# MobileNetV2 base (frozen initially)
cnn_base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3) if IMG_SIZE != 224 else (IMG_SIZE, IMG_SIZE, 3)
)
cnn_base.trainable = False

# Build model
input_layer = Input(shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3))

# Preprocessing
x = TimeDistributed(resize_layer)(input_layer)  # Resize if needed
x = TimeDistributed(BatchNormalization())(x)
x = TimeDistributed(RandomFlip("horizontal"))(x)
x = TimeDistributed(RandomZoom(0.05))(x)

# Feature extraction
x = TimeDistributed(cnn_base)(x)
x = TimeDistributed(GlobalAveragePooling2D())(x)

# Temporal modeling
x = LSTM(64, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)

# Classification head
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

# ===== TRAINING =====
# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Compile with gradient clipping
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

print("\nStarting training...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=callbacks,
    class_weight=class_weights
)

# ===== FINE-TUNING =====
print("\nStarting fine-tuning...")
cnn_base.trainable = True
for layer in cnn_base.layers[:-15]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001, clipnorm=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=8,
    callbacks=callbacks,
    class_weight=class_weights
)

# ===== SAVE RESULTS =====
model.save('sign_language_final.h5')
with open('class_labels.json', 'w') as f:
    json.dump(classes1.tolist(), f)

# Plot training
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'])
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'])
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.savefig('training_history.png')
plt.show()

print("\n✅ Training complete!")
print(f"Final Training Accuracy: {max(history.history['accuracy'] + history_fine.history['accuracy']):.4f}")
print(f"Final Validation Accuracy: {max(history.history['val_accuracy'] + history_fine.history['val_accuracy']):.4f}")
