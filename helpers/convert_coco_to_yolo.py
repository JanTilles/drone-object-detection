import os

import json

from tqdm import tqdm



# --- Configuration ---

BASE_DIR = "../dataset"

ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations")

IMAGE_SETS = ["train", "val", "test"]

OUTPUT_LABEL_DIR = os.path.join(BASE_DIR, "labels")



# Make sure label subfolders exist

for split in IMAGE_SETS:

    os.makedirs(os.path.join(OUTPUT_LABEL_DIR, split), exist_ok=True)



def convert_coco_to_yolo(ann_file, output_dir):

    with open(ann_file, "r") as f:

        data = json.load(f)



    # Map image_id to file_name and dimensions

    image_info = {img["id"]: img for img in data["images"]}



    for ann in tqdm(data["annotations"], desc=f"Converting {os.path.basename(ann_file)}"):

        img_id = ann["image_id"]

        category_id = ann["category_id"]

        bbox = ann["bbox"]  # COCO format: [x_min, y_min, width, height]



        # Image info

        img = image_info[img_id]

        img_w, img_h = img["width"], img["height"]

        file_name = os.path.splitext(img["file_name"])[0] + ".txt"



        # Normalize bbox to YOLO format: [x_center, y_center, width, height]

        x, y, w, h = bbox

        x_center = (x + w / 2) / img_w

        y_center = (y + h / 2) / img_h

        w /= img_w

        h /= img_h



        # Write to label file

        label_path = os.path.join(output_dir, file_name)

        with open(label_path, "a") as label_file:

            label_file.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")



# Run for train, val, test

for split in IMAGE_SETS:

    json_path = os.path.join(ANNOTATION_DIR, f"{split}.json")

    label_path = os.path.join(OUTPUT_LABEL_DIR, split)

    convert_coco_to_yolo(json_path, label_path)



print("✅ Conversion completed successfully.")


