# For VSCode (make project root as cwd)
import sys
sys.path.append(r"/home/chenxuyang/PythonProjects/ASPECT")

import torch
import os
import numpy as np
import yaml
import tqdm
from ultralytics import YOLO
from torch import nn
from utils.tools import find_files_by_suffix
from model.refiner import PoseTransformer

class D4PEDYOLOFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, dataset_type, yolo_model, seq_len=5, data_type='.png', device='cuda:1'):
        with open(yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        self.dataset_type = dataset_type
        self.yolo_model = yolo_model
        self.data_root = self.yaml_data['path']
        self.data_dir = os.path.join(self.data_root, self.yaml_data[self.dataset_type])
        self.seq_len = seq_len
        self.data_type = data_type
        self.image_paths = self._find_image_paths()
        self.features = []
        self._hook_yolo()
        self.device = device
    
    def _hook_yolo(self):
        def hook_fn(module, input, output):
            self.features.append(output)
        self.yolo_model.model.model[15].register_forward_hook(hook_fn)
    
    def _find_image_paths(self):
        return find_files_by_suffix(self.data_dir, self.data_type)
    
    def _read_keypoints(self, image_path):
        label_path = os.path.splitext(image_path)[0] + '.txt'
        with open(label_path, 'r') as file:
            line = np.array(list(map(float, file.readline().strip().split())))
        raw_keypoints = line[5:]
        keypoints = raw_keypoints.reshape(-1, 2)
        return np.array(keypoints)
    
    def __len__(self):
        return len(self.image_paths) - self.seq_len + 1

    def __getitem__(self, idx):
        cur_image_path = self.image_paths[idx+self.seq_len-1]
        keypoints = self._read_keypoints(cur_image_path)
        # Convert keypoints to float32 tensor
        keypoints = torch.from_numpy(keypoints).float()
        
        features = []
        for i in range(self.seq_len):
            image_path = self.image_paths[idx+i]
            self.features = []
            results = self.yolo_model.predict(image_path, device=self.device, verbose=False, save=False)
            features.append(self.features[-1][0])
        return features, keypoints

def train():
    # params
    epochs = 10
    learning_rate = 1e-4
    batch_size = 16
    seq_length = 5
    device = torch.device('cuda:0')

    # save path
    save_dir_root = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner"
    project_name = "D4PED_refiner_1_train_1"
    best_model_save_path = os.path.join(save_dir_root, project_name, "best.pt")
    tensorboard_dir_name = "tensorboard"
    tensorboard_dir = os.path.join(save_dir_root, project_name, tensorboard_dir_name)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    else:
        print(f"Tensorboard directory {tensorboard_dir} already exists.")
        print("Delete and recreate it? (y/n)")
        print(">> ", end="")
        choice = input()
        if choice.lower() == 'y':
            os.rmdir(tensorboard_dir)
            os.makedirs(tensorboard_dir)
        else:
            print("Exit training.")
            exit(0)

    # load model
    model = PoseTransformer()
    yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_dynamics_train_v1_finetune_0/weights/best.pt')
    dataset = D4PEDYOLOFeatureDataset('/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_dynamics_v12_split_aug_2/dongfanghong_4_coco.yaml', 'train', yolo_model, device="cuda:1")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    pg_bar = tqdm.tqdm(range(epochs))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in dataloader:
            features, keypoints = batch
            keypoints = keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pg_bar.set_postfix({'batch_loss': loss.item(), 'epoch_loss': train_loss})
        
        # average loss
        avg_loss = train_loss / len(dataloader)
        pg_bar.update(1)

    torch.save(model.state_dict(),best_model_save_path)

if __name__ == '__main__':
    train()
    
    