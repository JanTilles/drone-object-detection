import os



def validate_yolo_pairs(image_dir, label_dir, image_exts={".jpg", ".jpeg", ".png"}):

    image_files = sorted([f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_exts])

    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])



    image_basenames = set(os.path.splitext(f)[0] for f in image_files)

    label_basenames = set(os.path.splitext(f)[0] for f in label_files)



    missing_labels = image_basenames - label_basenames

    missing_images = label_basenames - image_basenames



    print(f"✅ Total images: {len(image_basenames)}")

    print(f"✅ Total labels: {len(label_basenames)}")

    

    if missing_labels:

        print(f"❌ Images missing labels: {len(missing_labels)}")

        for f in sorted(missing_labels):

            print(f" - {f}.jpg")

    else:

        print("✅ All images have matching labels.")



    if missing_images:

        print(f"❌ Labels missing images: {len(missing_images)}")

        for f in sorted(missing_images):

            print(f" - {f}.txt")

    else:

        print("✅ All labels have matching images.")



# Example usage

validate_yolo_pairs(

    image_dir="/scratch/project_2013587/tillesja/DIANA/images/train",

    label_dir="/scratch/project_2013587/tillesja/DIANA/labels/train"

)


