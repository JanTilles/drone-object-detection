# class no 5 for docks
# Tasks:
# 1. read annotation files in dock-dataset folders
# 2. get the annotation file for the image in the DIANA dataset annotations folders
# 3. for each annotation file, read the image number and the bounding box coordinates
# 4. convert the box coordinates to COCO format
# 5. add the dock annotations to DIANA dataset annotations  file
# 6. save the annotation file (to new location)

import json
import os
import shutil


def find_sibling_id(data, target_file_name):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'file_name' and value == target_file_name:
                return data.get('id')
            elif isinstance(value, (dict, list)):
                result = find_sibling_id(value, target_file_name)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_sibling_id(item, target_file_name)
            if result is not None:
                return result
    return None


def convert_to_coco(dock_annotations_dir, diana_annotations_json, combined_annotations_json):

    with open(diana_annotations_json, 'r') as file:
            diana_train_data = json.load(file)

    categories = diana_train_data['categories']
    categories.append({'id': 5, 'name': 'dock', 'supercategory': 'dock'})
    annotations = diana_train_data['annotations']


    # List all files in the directory
    annotation_files = [f for f in os.listdir(dock_annotations_dir) if os.path.isfile(os.path.join(dock_annotations_dir, f))]

    # Process each annotation file
    for annotation_file in annotation_files:
        with open(os.path.join(dock_annotations_dir, annotation_file), 'r') as file:
            data = json.load(file)

        # Extract the key and boxes elements
        key = data['key']
        print(f"Key: {key}")
        sibling_id = find_sibling_id(diana_train_data, key)
        print(f"Sibling ID: {sibling_id}")

        boxes = data['boxes']

        # Convert the bounding box coordinates to COCO format
        print("Boxes:")

        new_id = 173002


        coco_data =[]
        for box in boxes:
            print(box)
            x = int(float(box['x']))
            y = int(float(box['y']))
            w = int(float(box['width']))
            h = int(float(box['height']))
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2

            annotations.append({"id": new_id,"image_id": sibling_id,"category_id": 5,"bbox": [x1,y1,x2,y2],"area": w * h,"segmentation": [],"iscrowd": 0})
            new_id += 1

    with open(combined_annotations_json, 'w') as file:
            json.dump(diana_train_data, file, indent=4)
        


convert_to_coco('.\\dock-dataset\\annotations\\train\\', '.\\dataset\\annotations\\train.json' , '.\\dock-dataset\\combined-annotations\\train.json')
convert_to_coco('.\\dock-dataset\\annotations\\test\\', '.\\dataset\\annotations\\test.json' , '.\\dock-dataset\\combined-annotations\\test.json')
convert_to_coco('.\\dock-dataset\\annotations\\val\\', '.\\dataset\\annotations\\val.json' , '.\\dock-dataset\\combined-annotations\\val.json')

