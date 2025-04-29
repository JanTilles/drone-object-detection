import json

import os

import shutil



# ============================

# CONFIGURATION

# ============================

SCRATCH_PATH = "../dataset"

IMAGE_PATH = os.path.join(SCRATCH_PATH, "images")  # Source folder (contains all images)

ANNOTATION_PATH = os.path.join(SCRATCH_PATH, "annotations")



# Destination folders

TRAIN_PATH = os.path.join(IMAGE_PATH, "train")

VAL_PATH = os.path.join(IMAGE_PATH, "val")

TEST_PATH = os.path.join(IMAGE_PATH, "test")



# Ensure target directories exist

os.makedirs(TRAIN_PATH, exist_ok=True)

os.makedirs(VAL_PATH, exist_ok=True)

os.makedirs(TEST_PATH, exist_ok=True)



# ============================

# FUNCTION TO MOVE IMAGES

# ============================

def move_images(annotation_file, target_folder):

    """Moves images into their respective train, val, or test folders based on COCO annotation files."""

    with open(annotation_file, "r") as f:

        data = json.load(f)



    image_filenames = [img["file_name"] for img in data["images"]]



    for img_name in image_filenames:

        src = os.path.join(IMAGE_PATH, img_name)

        dst = os.path.join(target_folder, img_name)



        if os.path.exists(src):

            shutil.move(src, dst)

            print(f"✅ Moved {img_name} to {target_folder}")

        else:

            print(f"⚠️ WARNING: {img_name} not found in {IMAGE_PATH}")



# ============================

# MOVE IMAGES FOR TRAIN/VAL/TEST

# ============================

move_images(os.path.join(ANNOTATION_PATH, "train.json"), TRAIN_PATH)

move_images(os.path.join(ANNOTATION_PATH, "val.json"), VAL_PATH)

move_images(os.path.join(ANNOTATION_PATH, "test.json"), TEST_PATH)



print("✅ Image splitting completed successfully!")


