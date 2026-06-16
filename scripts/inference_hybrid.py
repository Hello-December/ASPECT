# for VSCode only
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import os
import time
import torch
import cv2
import numpy as np
import pandas as pd
import tqdm
import scipy
import shutil
from scipy.spatial.transform import Rotation
from ultralytics import YOLO
from model.refiner import PoseTransformer
from utils.tools import find_files_by_suffix
from configs.d4ped_camera import D4PED_CAMERA_INTRINSIC, D4PED_DIST_COEFFS

class ImageQuery:
    def __init__(self, image_paths_list, pose_model, yolo_model, camera_intrinsic, camera_dist_coeffs, init_3d_points, kpts_num=8, device="cuda:2"):
        self.image_paths_list = image_paths_list
        self.pose_model = pose_model
        self.yolo_model = yolo_model
        self.device = device
        self.torch_device = torch.device(self.device)
        self.features = []
        self.pose_model.to(self.torch_device)
        self.hook_yolo()
        self.cached_features = {}
        self.kpts_num = kpts_num
        self.camera_intrinsic = camera_intrinsic
        self.camera_dist_coeffs = camera_dist_coeffs
        self.init_3d_points = init_3d_points

    def hook_yolo(self):
        def hook_fn(module, input, output):
            self.features.append(output)
        self.yolo_model.model.model[15].register_forward_hook(hook_fn)

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image, image.shape[0], image.shape[1]

    def get_yolo_feature(self, image):
        self.features = []
        result = self.yolo_model(image, verbose=False, save=False, device=self.device)
        return self.features[-1][0], result

    def __len__(self):
        return len(self.image_paths_list)

    def simply_get_keypoints(self, idx, sequence_len=1):
        if idx + sequence_len > len(self.image_paths_list):
            raise IndexError(f"Index {idx + sequence_len - 1} in sequence out of range")
        features = []
        for i in range(sequence_len):
            image, _, _ = self.load_image(self.image_paths_list[idx + i])
            feature, _ = self.get_yolo_feature(image).cpu().numpy()
            features.append(feature)
        features = torch.tensor(np.array(features), device=self.torch_device)
        keypoints = self.pose_model(features).detach().cpu().numpy()
        return keypoints, (idx+sequence_len-1)

    def get_keypoints_cached(self, idx, sequence_len=1):
        FEATURE_CACHE_LEN = sequence_len + 5
        if idx > len(self.image_paths_list):
            raise IndexError(f"Index {idx} in sequence out of range")
        # judge if idx - sequence_len < 0, yes then detected by YOLO
        should_yolo_detect = idx - sequence_len + 1 < 0
        if should_yolo_detect:
            this_image, _, _ = self.load_image(self.image_paths_list[idx])
            feature, yolo_result = self.get_yolo_feature(this_image)
            result = yolo_result[0].keypoints.xyn.cpu().numpy().astype(np.float32).squeeze()
            self.cached_features[idx] = feature.cpu()
        else:
            features = []
            sequence_key_query = self.cached_features.keys()
            # print(sequence_key_query)
            # print(self.cached_features)
            for i in range(idx - sequence_len + 1, idx + 1):
                if i not in sequence_key_query:
                    this_image, _, _ = self.load_image(self.image_paths_list[i])
                    feature, _ = self.get_yolo_feature(this_image)
                    self.cached_features[i] = feature.cpu()
                    if i != idx:
                        print(f"Cached {i} not found.")
                features.append(self.cached_features[i])
            features = torch.tensor(np.array(features), device=self.torch_device)
            result = self.pose_model(features).detach().cpu().numpy().squeeze()
        # remove extra feature cache
        sorted_cached_features_keys = sorted(self.cached_features.keys())
        cached_features_keys_to_remove = sorted_cached_features_keys[0:-FEATURE_CACHE_LEN]
        if cached_features_keys_to_remove:
            for i in cached_features_keys_to_remove:
                del self.cached_features[i]
        return result

    def get_pose_hybrid(self, idx, sequence_len=1):
        FEATURE_CACHE_LEN = sequence_len + 5
        if idx > len(self.image_paths_list):
            raise IndexError(f"Index {idx} in sequence out of range")
        this_image, image_height, image_width = self.load_image(self.image_paths_list[idx])
        feature, yolo_result = self.get_yolo_feature(this_image)
        keypoints = yolo_result[0].keypoints.xyn.cpu().numpy().astype(np.float32).squeeze()
        self.cached_features[idx] = feature.cpu()
        if len(keypoints) < self.kpts_num:
            success = False
        else:
            keypoints_cv2 = keypoints.copy()
            keypoints_cv2[:, 0] *= image_width
            keypoints_cv2[:, 1] *= image_height
            keypoints_cv2[:, 1] = image_height - keypoints_cv2[:, 1]  # [IMPORTANT] convert to cv2 format (which is y-down, not y-up)
            (success, pred_rot_vec, pred_trans_vec) = cv2.solvePnP(self.init_3d_points, keypoints_cv2, self.camera_intrinsic, self.camera_dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not success:
            features = []
            sequence_key_query = self.cached_features.keys()
            # print(sequence_key_query)
            # print(self.cached_features)
            for i in range(idx - sequence_len + 1, idx + 1):
                if i not in sequence_key_query:
                    this_image, _, _ = self.load_image(self.image_paths_list[i])
                    feature, _ = self.get_yolo_feature(this_image)
                    self.cached_features[i] = feature.cpu()
                    if i != idx:
                        print(f"Cached {i} not found.")
                features.append(self.cached_features[i])
            features = torch.tensor(np.array(features), device=self.torch_device)
            keypoints = self.pose_model(features).detach().cpu().numpy().squeeze()
            keypoints_cv2 = keypoints.copy()
            keypoints_cv2[:, 0] *= image_width
            keypoints_cv2[:, 1] *= image_height
            keypoints_cv2[:, 1] = image_height - keypoints_cv2[:, 1]  # [IMPORTANT] convert to cv2 format (which is y-down, not y-up)
            (_, pred_rot_vec, pred_trans_vec) = cv2.solvePnP(self.init_3d_points, keypoints_cv2, self.camera_intrinsic, self.camera_dist_coeffs, flags=cv2.SOLVEPNP_EPNP)

        # cache management
        sorted_cached_features_keys = sorted(self.cached_features.keys())
        cached_features_keys_to_remove = sorted_cached_features_keys[0:-FEATURE_CACHE_LEN]
        if cached_features_keys_to_remove:
            for i in cached_features_keys_to_remove:
                del self.cached_features[i]
        return keypoints_cv2, pred_rot_vec, pred_trans_vec

def detect_pose():
    # WRITE_DIR = os.path.join("./files/rot_speed_analysis/test_hybrid_v1.1_train_1_dfh_4_speedplus_v4_aug_2")
    WRITE_DIR = os.path.join("./files/rot_speed_analysis/test_hybrid_v1.1_latency_train_3_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12")
    SEQ_LEN = 5
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    pred_rot_euler_cache = []
    pred_rot_quat_cache = []
    pred_rot_mat_cache = []
    pred_kpts_cache = []
    pred_latency_cache = []
    yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_dynamics_train_v1_finetune_0/weights/best.pt')
    # yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_train_v1_finetune_0/weights/best.pt')
    pose_model_path = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner/D4PED_dynamics_refiner_v1.1_train_3_direct/best_val_dict.pt"
    # pose_model_path = "/home/chenxuyang/PythonProjects/ASPECT/files/refiner/D4PED_speedplus_refiner_v1.1_train_1_direct/best_val_dict.pt"
    init_3d_points_label_path = PROJECT_ROOT / "assets" / "keypoints" / "00000.mat"
    camera_intrinsic = D4PED_CAMERA_INTRINSIC.copy()
    dist_coeffs = D4PED_DIST_COEFFS.copy()
    init_3d_points = scipy.io.loadmat(str(init_3d_points_label_path))["vertices_world_coords"]
    init_3d_points = init_3d_points.T.astype(np.float32).squeeze()[:8]  # only use first 8 points
    pose_model = PoseTransformer()
    pose_model.to(torch.device(device))
    pose_model.load_state_dict(torch.load(pose_model_path, map_location=torch.device(device)))
    pose_model.eval()
    img_dir = r"/home/chenxuyang/datasets/103_spacecraft/v2/original/dfh_4_dynamics_v12/random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12"
    # img_dir = r"/home/chenxuyang/datasets/103_spacecraft/v2/dfh_4_speedplus_v4_aug_2/test"
    img_paths = find_files_by_suffix(img_dir, ".png")
    img_paths.sort()
    image_query = ImageQuery(img_paths, pose_model, yolo_model, camera_intrinsic, dist_coeffs, init_3d_points, kpts_num=8, device=device)
    pg_bar = tqdm.tqdm(range(len(image_query)))
    pg_bar.set_description(f"Processing")
    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)
    else:
        print(f"Warning: {WRITE_DIR} already exists.")
        print("Delete and recreate? (y/n)")
        print(">> ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(WRITE_DIR)
            os.makedirs(WRITE_DIR)
        else:
            print("Exit.")
            exit(0)
    pred_rot_euler_cache_path = os.path.join(WRITE_DIR, "pred_rot_euler_cache.npy")
    pred_rot_quat_cache_path = os.path.join(WRITE_DIR, "pred_rot_quat_cache.npy")
    pred_rot_mat_cache_path = os.path.join(WRITE_DIR, "pred_rot_mat_cache.npy")
    pred_kpts_cache_path = os.path.join(WRITE_DIR, "pred_kpts_cache.npy")
    pred_latency_xlsx_path = os.path.join(WRITE_DIR, "pred_latency_cache.xlsx")

    for idx in pg_bar:
        torch.cuda.synchronize()
        t_start = time.time()
        keypoints, pred_rot_vec, _ = image_query.get_pose_hybrid(idx, SEQ_LEN)
        torch.cuda.synchronize()
        t_end = time.time()
        pred_latency_cache.append(t_end - t_start)
        pred_rot_mat = cv2.Rodrigues(pred_rot_vec)[0]
        pred_rot_euler = Rotation.from_matrix(pred_rot_mat).as_euler("xyz", degrees=False)
        pred_rot_quat = Rotation.from_matrix(pred_rot_mat).as_quat(scalar_first=True)
        pred_rot_euler_cache.append(pred_rot_euler)
        pred_rot_quat_cache.append(pred_rot_quat)
        pred_rot_mat_cache.append(pred_rot_mat)
        pred_kpts_cache.append(keypoints)
    
    np.save(pred_rot_euler_cache_path, np.array(pred_rot_euler_cache))
    np.save(pred_rot_quat_cache_path, np.array(pred_rot_quat_cache))
    np.save(pred_rot_mat_cache_path, np.array(pred_rot_mat_cache))
    np.save(pred_kpts_cache_path, np.array(pred_kpts_cache))

    latency_df = pd.DataFrame({
        "frame_idx": np.arange(len(pred_latency_cache)),
        "latency_s": pred_latency_cache,
        "latency_ms": np.array(pred_latency_cache) * 1000
    })
    avg_latency = latency_df["latency_s"].mean()
    latency_df.to_excel(pred_latency_xlsx_path, index=False)
    print(f"\nAverage inference latency: {avg_latency*1000:.2f} ms per frame")
    print(f"Total frames: {len(pred_latency_cache)}")
    print(f"Latency saved to: {pred_latency_xlsx_path}")

if __name__ == "__main__":
    detect_pose()

       