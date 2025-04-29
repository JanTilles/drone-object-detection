import os
import random
import cv2
from ultralytics import YOLO, settings

def load_yolo_labels(label_file):
    """Load YOLO labels from a file."""
    with open(label_file, 'r') as f:
        labels = []
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            labels.append((class_id, x_center, y_center, width, height))
        return labels

def draw_bounding_boxes(image, labels, class_names):
    """Draw bounding boxes and labels on the image."""
    h, w, _ = image.shape
    for label in labels:
        class_id, x_center, y_center, width, height = label
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(image_dir):
    """Main function to display random images with bounding boxes."""
    # Load class names
    #with open(class_names_file, 'r') as f:
    #    class_names = [line.strip() for line in f.readlines()]

    model = YOLO("../best.pt") # Load pre-trained YOLO model

    # Get list of images and labels
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    #labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]

    for i, idx in enumerate(random.sample(range(len(images)), min(5, len(images)))):
        # Choose a random image
        image_path = os.path.join(image_dir, images[idx])

        #results = model.predict(source=image_path,  show_labels=False)
        results = model.predict( source=image_path, show_labels=False, show_conf=False)

        for j, result in enumerate(results):
            # Add file name to the upper left corner
            #image = result.orig_img
            #cv2.putText(image, images[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            result.save(filename=f"result_{i}.jpg")

if __name__ == "__main__":
    # Replace these paths with your actual directories and class names file
    image_directory = "../dataset/images/val"

    main(image_directory)