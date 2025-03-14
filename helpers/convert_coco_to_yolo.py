from ultralytics.data.converter import convert_coco
from pathlib import Path
import json
import shutil


# Define paths
BASE_DIR = Path(".")  # Root directory (drone-object-detection/)
ANNOTATIONS_DIR = BASE_DIR / "annotations"  # Original COCO annotations
IMAGES_DIR = BASE_DIR / "images"  # Original images
DATASET_DIR = BASE_DIR / "dataset"  # New dataset structure
LABELS_DIR = DATASET_DIR / "labels"  # YOLO labels go here


def clean_annotation_folder():
    """Delete unwanted annotation JSON files, keeping only train/val/test.json."""
    print("Cleaning annotation folder...")

    for json_file in ANNOTATIONS_DIR.glob("*.json"):
        if json_file.name not in {"train.json", "val.json", "test.json"}:
            json_file.unlink()  # Delete unwanted JSON files
            print(f"Deleted {json_file.name}")


def increment_category_ids():
    """Increment category and annotation IDs in COCO annotation files."""
    print("Incrementing category IDs in COCO annotations...")
    
    for split in ["train", "val", "test"]:
        annotation_file = ANNOTATIONS_DIR / f"{split}.json"
        
        if annotation_file.exists():
            with annotation_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # Increment class IDs
            for category in data.get("categories", []):
                category["id"] += 1
            
            for annotation in data.get("annotations", []):
                annotation["category_id"] += 1

            # Save updated COCO annotations
            with annotation_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            print(f"Updated IDs in {split}.json")


def convert_annotations():
    """Convert COCO annotations to YOLO format inside `dataset/labels/`."""
    print("Starting COCO to YOLO conversion...")

    convert_coco(
        labels_dir=str(ANNOTATIONS_DIR),  # Convert all remaining JSONs
        save_dir=str(DATASET_DIR),
        use_segments=False,
        use_keypoints=False,
        cls91to80=False,
        lvis=False
    )

    print("COCO annotations successfully converted to YOLO format.")


def move_images():
    """Move images into their respective train/val/test folders inside `dataset/images/`."""
    print("Moving images into dataset/images/...")

    for split in ["train", "val", "test"]:
        annotation_file = ANNOTATIONS_DIR / f"{split}.json"
        image_dest = DATASET_DIR / "images" / split  # New image storage location
        
        # Ensure subdirectory exists
        image_dest.mkdir(parents=True, exist_ok=True)
        
        if annotation_file.exists():
            with annotation_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            total_images = len(data.get("images", []))
            print(f"Processing {total_images} images for {split}...")

            for idx, image in enumerate(data.get("images", []), start=1):
                image_path = IMAGES_DIR / image["file_name"]
                if image_path.exists():
                    shutil.move(str(image_path), str(image_dest / image["file_name"]))
                if idx % 1000 == 0:
                    print(f"Moved {idx}/{total_images} images for {split}...")


def cleanup():
    """Delete the original annotations and images folders."""
    print("Cleaning up...")

    # Delete original annotations directory
    if ANNOTATIONS_DIR.exists():
        shutil.rmtree(ANNOTATIONS_DIR)
        print("Deleted original annotations folder.")

    # Delete original images directory
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
        print("Deleted original images folder.")


def main():
    """Main function to orchestrate dataset preparation."""
    clean_annotation_folder()  # Step 1: Remove unwanted JSON files
    increment_category_ids()   # Step 2: Update class IDs
    convert_annotations()      # Step 3: Convert COCO to YOLO
    move_images()              # Step 4: Move images
    cleanup()                  # Step 5: Cleanup
    print("COCO to YOLO conversion and image organization complete.")


if __name__ == "__main__":
    main()
