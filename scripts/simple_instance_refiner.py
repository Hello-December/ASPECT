# for VSCode only
import sys
sys.path.append("/home/chenxuyang/PythonProjects/ASPECT/")

import os
import torch
import cv2
import numpy as np
import tqdm
from ultralytics import YOLO
from model.refiner import PoseTransformer
from utils.tools import find_files_by_suffix

class ImageQuery:
    def __init__(self, image_paths_list, pose_model, yolo_model, device="cuda:2"):
        self.image_paths_list = image_paths_list
        self.pose_model = pose_model
        self.yolo_model = yolo_model
        self.device = device
        self.torch_device = torch.device(self.device)
        self.features = []
        self.pose_model.to(self.torch_device)
        self.hook_yolo()

    def hook_yolo(self):
        def hook_fn(module, input, output):
            self.features.append(output)
        self.yolo_model.model.model[15].register_forward_hook(hook_fn)

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def get_yolo_feature(self, image):
        self.features = []
        _ = self.yolo_model(image, verbose=False, save=False, device=self.device)
        return self.features[-1][0]

    def __len__(self):
        return len(self.image_paths_list)

    def get_keypoints(self, idx, sequence_len=1):
        if idx + sequence_len > len(self.image_paths_list):
            raise IndexError(f"Index {idx + sequence_len - 1} in sequence out of range")
        features = []
        for i in range(sequence_len):
            image = self.load_image(self.image_paths_list[idx + i])
            feature = self.get_yolo_feature(image).cpu().numpy()
            features.append(feature)
        features = torch.tensor(np.array(features), device=self.torch_device)
        keypoints = self.pose_model(features).detach().cpu().numpy()
        return keypoints, (idx+sequence_len-1)

def test_get_keypoints():
    WRITE_DIR = os.path.join("./files/instance/test_refiner_1/")
    SEQ_LEN = 5
    device = "cuda:2"
    os.makedirs(WRITE_DIR, exist_ok=True)
    yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_train_v1_finetune_0/weights/best.pt')
    pose_model_path = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner/D4PED_refiner_train_1_direct/model_epoch_60.pt"
    pose_model = PoseTransformer()
    pose_model.load_state_dict(torch.load(pose_model_path))
    pose_model.eval()
    img_dir = "./files/test"
    img_paths = find_files_by_suffix(img_dir, ".png")

    image_query = ImageQuery(img_paths, pose_model, yolo_model, device=device)
    max_idx = len(image_query) - SEQ_LEN
    pg_bar = tqdm.tqdm(range(max_idx))
    for idx in pg_bar:
        pg_bar.set_description(f"Processing image | ")
        keypoints, this_image_id = image_query.get_keypoints(idx, SEQ_LEN)

        this_image_path = image_query.image_paths_list[this_image_id]
        this_image_name = os.path.basename(this_image_path)
        save_image_name = os.path.join(WRITE_DIR, this_image_name)
        save_image = image_query.load_image(this_image_path)
        for i in range(keypoints.shape[0]):
            kp = keypoints[i]
            for j in range(kp.shape[0]):
                x, y = int(kp[j, 0] * save_image.shape[1]), int(kp[j, 1] * save_image.shape[0])
                cv2.circle(save_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(save_image_name, save_image)

if __name__ == "__main__":
    test_get_keypoints()
    
            