import os
# 1. Suppress harmless C++ PNG warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers, models, applications

# Dataset location
DATASET_BASE = r"C:\Users\ASUS\.cache\kagglehub\datasets\swoyam2609\fresh-and-stale-classification\versions\1"

def find_split_dir(base_path, target_names):
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            if d.lower() in target_names:
                return os.path.join(root, d)
    return base_path

train_dir = find_split_dir(DATASET_BASE, ["train"])
test_dir = find_split_dir(DATASET_BASE, ["test", "val", "validation"])

# 2. Find matching folder classes between train and val
train_subdirs = set(d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)))
test_subdirs = set(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
common_classes = sorted(list(train_subdirs.intersection(test_subdirs)))

print(f"Train directory: {train_dir}")
print(f"Validation directory: {test_dir}")
print(f"Synchronized classes ({len(common_classes)}): {common_classes}\n")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 3. Load datasets using common_classes
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    class_names=common_classes,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    class_names=common_classes,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

num_classes = len(common_classes)

# 4. Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# 5. Model Architecture
base_model = applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# Phase 1: Train Top Classifier
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("--- Phase 1: Training Classification Head ---")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Phase 2: Fine-Tune Top Layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Phase 2: Fine-Tuning Top Layers ---")
model.fit(train_ds, validation_data=val_ds, epochs=5)

# Save Trained Model
model.save("fruit_model.h5")
print("\nModel successfully saved as 'fruit_model.h5'!")