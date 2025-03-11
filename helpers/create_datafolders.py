import os

def create_data_folders(base_path):
    folders = [
        'images/train',
        'images/val',
        'images/test',  # Added test folder
        'labels/train',
        'labels/val',
        'labels/test'  # Added test folder
    ]
    
    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f'Created folder: {path}')
        
        # Create a .gitkeep file in each folder
        gitkeep_path = os.path.join(path, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            pass
        if os.path.exists(gitkeep_path):
            print(f'Created .gitkeep in: {gitkeep_path}')
        else:
            print(f'Failed to create .gitkeep in: {gitkeep_path}')

if __name__ == '__main__':
    base_path = 'c:/Users/extjtilles/Documents/Work/Python/drone-object-detection/dataset'
    create_data_folders(base_path)
