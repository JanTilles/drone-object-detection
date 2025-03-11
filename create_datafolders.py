import os

def create_data_folders(base_path):
    folders = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f'Created folder: {path}')

if __name__ == '__main__':
    base_path = 'c:/Users/extjtilles/Documents/Work/Python/drone-object-detection/dataset'
    create_data_folders(base_path)
