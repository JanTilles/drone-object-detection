import os
import random
import cv2

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

def main(image_dir, label_dir, class_names_file):
    """Main function to display random images with bounding boxes."""
    # Load class names
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Get list of images and labels
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    labels = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]

    while True:
        # Choose a random image
        idx = random.randint(0, len(images) - 1)
        image_path = os.path.join(image_dir, images[idx])
        label_path = os.path.join(label_dir, labels[idx])

        # Load image and labels
        image = cv2.imread(image_path)
        yolo_labels = load_yolo_labels(label_path)

        # Scale the image to half its size
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

        # Draw bounding boxes
        draw_bounding_boxes(image, yolo_labels, class_names)

        # Add image file name to the top-left corner
        cv2.putText(image, images[idx], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Show image
        cv2.imshow('Image with Bounding Boxes', image)

        # Wait for key press
        key = cv2.waitKey(0)
        if key == ord('n'):  # Press 'n' to see another image
            continue
        elif key == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace these paths with your actual directories and class names file
    image_directory = "../dataset/images/train"
    label_directory = "../dataset/labels/train"
    class_names_file = "../dataset/labels/categories.txt"

    main(image_directory, label_directory, class_names_file)