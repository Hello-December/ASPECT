# For VSCode (make project root as cwd)
import sys
sys.path.append(r"/home/chenxuyang/PythonProjects/ASPECT")

import torch
import os
import numpy as np
import yaml
import tqdm
import shutil
from ultralytics import YOLO
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from utils.tools import find_files_by_suffix
from model.refiner import PoseTransformer

class D4PEDYOLOPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_path, feature_root, dataset_type, seq_len=5, data_type='.png', device=torch.device('cpu'), preload=False):
        with open(yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)
        self.dataset_type = dataset_type
        self.feature_root = feature_root
        self.feature_dir = os.path.join(self.feature_root, self.dataset_type)
        self.data_root = self.yaml_data['path']
        self.data_dir = os.path.join(self.data_root, self.yaml_data[self.dataset_type])
        self.seq_len = seq_len
        self.data_type = data_type
        self.image_paths = self._find_image_paths()
        self.device = device
        self.preload = preload
        if self.preload:
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
    def __getitem__(self, idx):
        cur_image_path = self.image_paths[idx+self.seq_len-1]
        keypoints = self._read_keypoints(cur_image_path)
        # Convert keypoints to float32 tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float)
        
        features = []
        if self.preload:
            for i in range(self.seq_len):
                image_path = self.image_paths[idx+i]
                feature = self.features[image_path]
                features.append(feature)
        else:
            for i in range(self.seq_len):
                image_path = self.image_paths[idx+i]
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                feature_path = os.path.join(self.feature_dir, image_name + '.npy')
                feature = np.load(feature_path)
                features.append(feature)
        return torch.tensor(np.array(features), dtype=torch.float), keypoints

    # preload method
    # def __getitem__(self, idx):
    #     cur_image_path = self.image_paths[idx+self.seq_len-1]
    #     keypoints = self._read_keypoints(cur_image_path)
    #     # Convert keypoints to float32 tensor
    #     keypoints = torch.tensor(keypoints, dtype=torch.float)
        
    #     features = []
    #     for i in range(self.seq_len):
    #         image_path = self.image_paths[idx+i]
    #         feature = self.features[image_path]
    #         features.append(feature)
    #     return torch.tensor(np.array(features), dtype=torch.float), keypoints

def train():
    # params
    epochs = 100
    learning_rate = 1e-4
    batch_size = 16
    seq_length = 5
    num_workers = 16
    save_interval = 5
    device = torch.device('cuda:2')

    # save path
    save_dir_root = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner"
    project_name = "D4PED_speedplus_refiner_v1.1_train_1_direct"
    project_dir = os.path.join(save_dir_root, project_name)
    best_train_model_dict_save_path = os.path.join(project_dir, "best_train_dict.pt")
    best_val_model_dict_save_path = os.path.join(project_dir, "best_val_dict.pt")
    best_train_model__save_path = os.path.join(project_dir, "best_train.pt")
    best_val_model__save_path = os.path.join(project_dir, "best_val.pt")
    tensorboard_dir_name = "tensorboard"
    tensorboard_dir = os.path.join(project_dir, tensorboard_dir_name)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    else:
        print(f"Project directory {project_dir} already exists.")
        print("Delete and recreate it? (y/n)")
        print(">> ", end="")
        choice = input()
        if choice.lower() == 'y':
            shutil.rmtree(project_dir)
            os.makedirs(tensorboard_dir)
        else:
            print("Exit training.")
            exit(0)

    # load model
    model = PoseTransformer()

    # load dataset
    dataset_yaml_path = r"/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_speedplus_v4_aug_2/dongfanghong_4_coco.yaml"
    dataset_train = D4PEDYOLOPreprocessedDataset(dataset_yaml_path, '/tmp/d4ped_speedplus_features/', 'train', seq_len=seq_length, device=device, preload=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataset_val = D4PEDYOLOPreprocessedDataset(dataset_yaml_path, '/tmp/d4ped_speedplus_features/', 'val', seq_len=seq_length, device=device, preload=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(device)

    # loss, learning rate scheduler and optimizer
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # initialize tensorboard writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    pg_bar = tqdm.tqdm(range(epochs))
    pg_bar.set_description(f"⏱️ Total Progress")
    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_pg_bar = tqdm.tqdm(dataloader_train)
        train_pg_bar.set_description(f"🚅 Train Progress of Epoch {epoch}")
        for batch in dataloader_train:
            features, keypoints = batch
            features = features.to(device)
            keypoints = keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, keypoints) * 100
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            this_loss = loss.item() * 100
            train_loss += this_loss
            train_pg_bar.update(1)
        train_pg_bar.close()
        
        # average loss
        avg_train_loss = train_loss / len(dataloader_train)
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(project_dir, f"model_dict_epoch_{epoch + 1}.pt"))
            torch.save(model, os.path.join(project_dir, f"model_epoch_{epoch + 1}.pt"))
            print(f"✅ Model saved with loss: {avg_train_loss:.4f}")

        # write to tensorboard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_total', train_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # validate on val set and write to tensorboard
        model.eval()
        val_loss = 0.0
        val_pg_bar = tqdm.tqdm(dataloader_val)
        val_pg_bar.set_description(f"🛂 Val Progress of Epoch {epoch}")
        with torch.no_grad():
            for batch in dataloader_val:
                features, keypoints = batch
                features = features.to(device)
                keypoints = keypoints.to(device)
                outputs = model(features)
                loss = criterion(outputs, keypoints) * 100
                this_loss = loss.item() * 100
                val_loss += this_loss
                val_pg_bar.update(1)
        val_pg_bar.close()
        
        # average loss
        avg_val_loss = val_loss / len(dataloader_val)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Loss/val_total', val_loss, epoch)
        pg_bar.set_postfix({'epoch_avg_train_loss': avg_train_loss, 'epoch_avg_val_loss': avg_val_loss, 'epoch_total_train_loss': train_loss, 'epoch_total_val_loss': val_loss})
        
        # save train best model
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), best_train_model_dict_save_path)
            torch.save(model, best_train_model__save_path)
            print(f"✅ Best train model saved with loss: {best_train_loss:.4f}")

        # save val best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_val_model_dict_save_path)
            torch.save(model, best_val_model__save_path)
            print(f"✅ Best val model saved with loss: {best_val_loss:.4f}")
        
        scheduler.step()
        pg_bar.update(1)

    # close progress bar
    pg_bar.close()

    # close tensorboard writer
    writer.close()
    print("✅ Training completed and tensorboard writer closed.")

if __name__ == '__main__':
    train()
    
    