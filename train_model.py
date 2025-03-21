import os

import json

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt



# ============================

# CONFIGURATION

# ============================

IMG_HEIGHT, IMG_WIDTH = 512, 288

BATCH_SIZE = 32

EPOCHS = 100



SCRATCH_PATH = "/scratch/project_2013587/tillesja/DIANA"

IMAGE_PATH = os.path.join(SCRATCH_PATH, "images")

ANNOTATION_PATH = os.path.join(SCRATCH_PATH, "annotations")



# ============================

# DETECT GPUs AUTOMATICALLY (NO LIMIT IN PYTHON)

# ============================

gpus = tf.config.list_physical_devices('GPU')

if gpus:

    print(f"✅ Multi-GPU Training Enabled: {len(gpus)} GPUs Found!")

    strategy = tf.distribute.MirroredStrategy()

else:

    print("⚠️ WARNING: No GPUs detected! Using CPU only.")

    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")



# ============================

# DATA LOADER FUNCTION

# ============================

def parse_coco(annotation_file, img_folder):

    """Creates a TensorFlow Dataset from COCO JSON annotations."""

    with open(annotation_file, "r") as f:

        data = json.load(f)



    img_paths, labels = [], []

    for img_info in data["images"]:

        img_path = os.path.join(img_folder, img_info["file_name"])

        if not os.path.exists(img_path):

            continue

        img_paths.append(img_path)

        img_id = img_info["id"]

        label = next((ann["category_id"] for ann in data["annotations"] if ann["image_id"] == img_id), 0)

        labels.append(label)

    

    return img_paths, labels



def load_and_preprocess_image(img_path, label):

    """Loads and preprocesses an image for training."""

    img = tf.io.read_file(img_path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

    img = img / 255.0  # Normalize

    return img, label



def build_tf_dataset(annotation_file, img_folder, batch_size):

    """Builds an efficient TensorFlow dataset pipeline."""

    img_paths, labels = parse_coco(annotation_file, img_folder)

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset



# ============================

# LOAD DATASETS (STREAM FROM DISK INSTEAD OF RAM)

# ============================

train_ds = build_tf_dataset(os.path.join(ANNOTATION_PATH, "train.json"), os.path.join(IMAGE_PATH, "train"), BATCH_SIZE)

val_ds = build_tf_dataset(os.path.join(ANNOTATION_PATH, "val.json"), os.path.join(IMAGE_PATH, "val"), BATCH_SIZE)

test_ds = build_tf_dataset(os.path.join(ANNOTATION_PATH, "test.json"), os.path.join(IMAGE_PATH, "test"), BATCH_SIZE)



# ============================

# BUILD MODEL INSIDE MULTI-GPU STRATEGY

# ============================

with strategy.scope():

    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base_model.trainable = False  # Freeze base model



    x = layers.GlobalAveragePooling2D()(base_model.output)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation='softmax')(x)  # Adjust classes



    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),

                  metrics=['accuracy'])



print("✅ Multi-GPU Model Compiled Successfully!")



# ============================

# TRAINING WITH MULTIPLE GPUs

# ============================

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

checkpoint = ModelCheckpoint('/scratch/project_2013587/tillesja/best_model.keras', monitor='val_loss', save_best_only=True)



history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[early_stop, checkpoint]

)



# ============================

# EVALUATION & SAVE RESULTS

# ============================

val_loss, val_acc = model.evaluate(val_ds, verbose=0)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)

print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")



# Save training curves

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Train Accuracy')

plt.plot(history.history['val_accuracy'], label='Val Accuracy')

plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.title('Training vs Validation Accuracy')

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Train Loss')

plt.plot(history.history['val_loss'], label='Val Loss')

plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training vs Validation Loss')

plt.legend()



plt.savefig('/scratch/project_2013587/tillesja/training_curves.png')

plt.close()



# Save final trained model

model.save("/scratch/project_2013587/tillesja/diana_trained_model.keras")



print("✅ Model and training curves saved successfully.")


