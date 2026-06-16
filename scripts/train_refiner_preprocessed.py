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

class D4PEDYOLOPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, feature_root, dataset_type, seq_len=5, data_type='.png', device=torch.device('cpu')):
        with open(yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        self.dataset_type = dataset_type
        self.feature_root = feature_root
        self.feature_dir = os.path.join(self.feature_root, self.dataset_type)
        self.data_root = self.yaml_data['path']
        self.data_dir = os.path.join(self.data_root, self.yaml_data[self.dataset_type])
        self.seq_len = seq_len
        self.data_type = data_type
        self.image_paths = self._find_image_paths()[:5000]
        self.device = device
        self._preload_features()
    
    def _find_image_paths(self):
        return find_files_by_suffix(self.data_dir, self.data_type)

    def _preload_features(self):
        print("🤓☝️  Begin preload features...")
        self.features = {}
        preload_pg_bar = tqdm.tqdm(range(len(self.image_paths)))
        for image_path in self.image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            feature_path = os.path.join(self.feature_dir, image_name + '.npy')
            feature = np.load(feature_path)
            self.features[image_path] = feature
            preload_pg_bar.update(1)
        preload_pg_bar.close()
        print("✅ Preload features complete.")
    
    def _read_keypoints(self, image_path):
        label_path = os.path.splitext(image_path)[0] + '.txt'
        with open(label_path, 'r') as file:
            line = np.array(list(map(float, file.readline().strip().split())))
        raw_keypoints = line[5:]
        keypoints = raw_keypoints.reshape(-1, 2)
        return np.array(keypoints)
    
    def __len__(self):
        return len(self.image_paths) - self.seq_len + 1

    # bruteforce load
    # def __getitem__(self, idx):
    #     cur_image_path = self.image_paths[idx+self.seq_len-1]
    #     keypoints = self._read_keypoints(cur_image_path)
    #     # Convert keypoints to float32 tensor
    #     keypoints = torch.from_numpy(keypoints).float()
        
    #     features = []
    #     for i in range(self.seq_len):
    #         image_path = self.image_paths[idx+i]
    #         image_name = os.path.splitext(os.path.basename(image_path))[0]
    #         feature_path = os.path.join(self.feature_dir, image_name + '.npy')
    #         feature = np.load(feature_path)
    #         features.append(feature)
    #     return np.array(features), keypoints

    # preload method
    def __getitem__(self, idx):
        cur_image_path = self.image_paths[idx+self.seq_len-1]
        keypoints = self._read_keypoints(cur_image_path)
        # Convert keypoints to float32 tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float)
        
        features = []
        for i in range(self.seq_len):
            image_path = self.image_paths[idx+i]
            feature = self.features[image_path]
            features.append(feature)
        return torch.tensor(np.array(features), dtype=torch.float), keypoints

def train():
    # params
    epochs = 100
    learning_rate = 1e-4
    batch_size = 32
    seq_length = 5
    num_workers = 32
    device = torch.device('cuda:0')

    # save path
    save_dir_root = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner"
    project_name = "D4PED_refiner_train_1"
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
    dataset = D4PEDYOLOPreprocessedDataset('/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_dynamics_v12_split_aug_2/dongfanghong_4_coco.yaml', './files/d4ped_features', 'train', seq_len=seq_length, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model.to(device)

    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    pg_bar = tqdm.tqdm(range(epochs))
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in dataloader:
            features, keypoints = batch
            features = features.to(device)
            keypoints = keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*20
            pg_bar.set_postfix({'batch_loss': loss.item(), 'epoch_loss': train_loss})
        
        # average loss
        avg_loss = train_loss / len(dataloader)
        pg_bar.update(1)

        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),best_model_save_path)
            print(f"✅ Best model saved with loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()
    
    