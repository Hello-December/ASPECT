# For VSCode (make project root as cwd)
import sys
sys.path.append(r"/home/chenxuyang/PythonProjects/ASPECT")

from ultralytics import YOLO
import torch
import numpy as np
import cv2
import tqdm
import os
import yaml
import shutil
from utils.tools import find_files_by_suffix

class D4PEDCOCOFormatDataset:
    def __init__(self, yaml_path, dataset_type, data_type='.png'):
        with open(yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        self.dataset_type = dataset_type
        self.data_root = self.yaml_data['path']
        self.data_dir = os.path.join(self.data_root, self.yaml_data[self.dataset_type])
        self.data_type = data_type
        self.image_paths = self._find_image_paths()

    def _find_image_paths(self):
        return find_files_by_suffix(self.data_dir, self.data_type)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, image_name

    def __len__(self):
        return len(self.image_paths)

def preprocess_yolo_features():
    yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_train_v1_finetune_0/weights/best.pt')
    dataset = D4PEDCOCOFormatDataset('/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_speedplus_v4_aug_2/dongfanghong_4_coco.yaml', 'train')
    pg_bar = tqdm.tqdm(range(len(dataset)))
    save_dir = f"/tmp/d4ped_speedplus_features/{dataset.dataset_type}"
    raw_features = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Warning: {save_dir} already exists.")
        print("Overwrite? (y/n)")
        print(">> ", end="")
        overwrite = input()
        if overwrite.lower() == 'y':
            print("Overwrite.")
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            print("Cancel.")
            return
        
    def hook_fn(module, input, output):
        raw_features.append(output)
    yolo_model.model.model[15].register_forward_hook(hook_fn)

    print(f"🤓☝️  Begin preprocess yolo features for {dataset.dataset_type} and save to {save_dir}...")
    for img, image_name in dataset:
        raw_features = []
        _ = yolo_model.predict(img, device='cuda:1', verbose=False, save=False)
        features = raw_features[-1][0]
        features = features.cpu().numpy()
        np.save(os.path.join(save_dir, image_name + '.npy'), features)
        pg_bar.update(1)
    pg_bar.close()
    print("✅ Preprocess yolo features complete.")

if __name__ == '__main__':
    preprocess_yolo_features()