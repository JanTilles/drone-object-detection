from ultralytics import YOLO

import matplotlib.pyplot as plt

import cv2

import os



# ============================

# CONFIGURATION

# ============================

MODEL_ARCH = "yolov8n.pt"  # Can be changed to yolov8s.pt, yolov8m.pt, etc.

EPOCHS = 100

IMG_SIZE = (960, 544)  # Keep original 16:9 image ratio

BATCH_SIZE = 16

PROJECT_DIR = "/scratch/project_2013587/tillesja/DIANA"

DATA_YAML = os.path.join(PROJECT_DIR, "data.yaml")

SAVE_MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")



# ============================

# TRAINING

# ============================

print("\n🚀 Starting training with image size:", IMG_SIZE)

model = YOLO(MODEL_ARCH)

model.train(

    data=DATA_YAML,

    epochs=EPOCHS,

    imgsz=IMG_SIZE,

    batch=BATCH_SIZE,

    project=os.path.join(PROJECT_DIR, "runs/train"),

    name="exp",

    exist_ok=True,

    save=True,

    verbose=True,

    patience=20,  # Early stopping

)



# ============================

# VALIDATION

# ============================

print("\n✅ Validating the best model...")



val_model = YOLO(SAVE_MODEL_PATH)



val_results = val_model.val(data=DATA_YAML, split='val')

print(val_results)



# ============================

# PLOT SAMPLE VALIDATION IMAGE

# ============================

print("\n🖼️ Generating a prediction on a validation image...")

val_image_dir = os.path.join(PROJECT_DIR, "images/val")

val_images = [f for f in os.listdir(val_image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]



if val_images:

    sample_image_path = os.path.join(val_image_dir, val_images[0])

    results = model.predict(source=sample_image_path, conf=0.25, save=False)

    result_img = results[0].plot()



    # Convert BGR to RGB for matplotlib

    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    plt.imshow(result_img_rgb)

    plt.axis('off')

    plt.title("Sample Validation Prediction")

    plt.show()

else:

    print("⚠️ No validation images found in:", val_image_dir)



# ============================

# DONE

# ============================

print("\n🎉 Training complete! Best model and training logs are saved in 'runs/train/exp'")


