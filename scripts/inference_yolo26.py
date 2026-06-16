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
from utils.tools import find_files_by_suffix
from configs.d4ped_camera import D4PED_CAMERA_INTRINSIC, D4PED_DIST_COEFFS

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, image.shape[0], image.shape[1]

def inference_yolo():
    DEVICE = "cuda:0"
    WRITE_DIR = os.path.join("./files/rot_speed_analysis/test_yolo26s_latency_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/")
    # img_dir = "/home/chenxuyang/datasets/103_spacecraft/v2/original/speedplus_v4/synthetic_speedplusv2_dfh-4_test_test1_v10"
    # img_dir = "/home/chenxuyang/datasets/103_spacecraft/v2/original/dfh_4_dynamics_v12/random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12"
    # img_dir = r"/home/chenxuyang/datasets/103_spacecraft/v2/original/dfh_4_dynamics_v12/random_light_0.1_0.7_0.08_0.1_1000_v2_2000imgs_v12"
    img_dir = r"/home/chenxuyang/datasets/103_spacecraft/v2/original/dfh_4_dynamics_v12/random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12"
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

    # yolo_model = YOLO('/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_train_v1_finetune_0/weights/best.pt')
    yolo_model = YOLO("/home/chenxuyang/PythonProjects/ASPECT/runs/pose/models/D4PED/yolo26s_D4PED_dynamics_train_v1_finetune_0/weights/best.pt")
    init_3d_points_label_path = PROJECT_ROOT / "assets" / "keypoints" / "00000.mat"
    camera_intrinsic = D4PED_CAMERA_INTRINSIC.copy()
    dist_coeffs = D4PED_DIST_COEFFS.copy()
    init_3d_points = scipy.io.loadmat(str(init_3d_points_label_path))["vertices_world_coords"]
    init_3d_points = init_3d_points.T.astype(np.float32).squeeze()[:8]  # only use first 8 points

    pred_rot_euler_cache_path = os.path.join(WRITE_DIR, "pred_rot_euler_cache.npy")
    pred_rot_quat_cache_path = os.path.join(WRITE_DIR, "pred_rot_quat_cache.npy")
    pred_rot_mat_cache_path = os.path.join(WRITE_DIR, "pred_rot_mat_cache.npy")
    pred_kpts_cache_path = os.path.join(WRITE_DIR, "pred_kpts_cache.npy")
    pred_latency_xlsx_path = os.path.join(WRITE_DIR, "pred_latency_cache.xlsx")

    pred_rot_euler_cache = []
    pred_rot_quat_cache = []
    pred_rot_mat_cache = []
    pred_kpts_cache = []
    pred_latency_cache = []

    prev_rot_euler = None
    prev_rot_quat = None
    prev_rot_mat = None
    prev_kpts = None

    img_paths = find_files_by_suffix(img_dir, ".png")
    img_paths.sort()
    for img_path in tqdm.tqdm(img_paths):
        image, img_height, _ = load_image(img_path)
        torch.cuda.synchronize()
        t_start = time.time()
        yolo_result = yolo_model.predict(image, verbose=False, save=False, device=DEVICE)
        torch.cuda.synchronize()
        t_end = time.time()
        pred_latency_cache.append(t_end - t_start)
        keypoints = yolo_result[0].keypoints.xy.cpu().numpy().astype(np.float32).squeeze()
        
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(1, -1)
        
        if len(keypoints) >= 4:
            pred_keypoints_cv2 = keypoints.copy()
            for this_pred_keypoint in pred_keypoints_cv2:
                this_pred_keypoint[1] = img_height - this_pred_keypoint[1]
            (success, pred_rot_vec, _) = cv2.solvePnP(init_3d_points, pred_keypoints_cv2, camera_intrinsic, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            if success:
                pred_rot_mat = cv2.Rodrigues(pred_rot_vec)[0]
                pred_pose_euler = Rotation.from_matrix(pred_rot_mat).as_euler("xyz", degrees=False)
                pred_pose_quat = Rotation.from_matrix(pred_rot_mat).as_quat(scalar_first=True)
                prev_rot_euler = pred_pose_euler
                prev_rot_quat = pred_pose_quat
                prev_rot_mat = pred_rot_mat
                prev_kpts = keypoints
                pred_rot_euler_cache.append(pred_pose_euler)
                pred_rot_quat_cache.append(pred_pose_quat)
                pred_rot_mat_cache.append(pred_rot_mat)
                pred_kpts_cache.append(keypoints)
            else:
                if prev_rot_euler is not None:
                    pred_rot_euler_cache.append(prev_rot_euler)
                    pred_rot_quat_cache.append(prev_rot_quat)
                    pred_rot_mat_cache.append(prev_rot_mat)
                    pred_kpts_cache.append(prev_kpts)
                else:
                    pred_rot_euler_cache.append(np.zeros(3))
                    pred_rot_quat_cache.append(np.array([1.0, 0.0, 0.0, 0.0]))
                    pred_rot_mat_cache.append(np.eye(3))
                    pred_kpts_cache.append(np.zeros((8, 2)))
        else:
            if prev_rot_euler is not None:
                pred_rot_euler_cache.append(prev_rot_euler)
                pred_rot_quat_cache.append(prev_rot_quat)
                pred_rot_mat_cache.append(prev_rot_mat)
                pred_kpts_cache.append(prev_kpts)
            else:
                pred_rot_euler_cache.append(np.zeros(3))
                pred_rot_quat_cache.append(np.array([1.0, 0.0, 0.0, 0.0]))
                pred_rot_mat_cache.append(np.eye(3))
                pred_kpts_cache.append(np.zeros((8, 2)))
    
    np.save(pred_rot_euler_cache_path, pred_rot_euler_cache)
    np.save(pred_rot_quat_cache_path, pred_rot_quat_cache)
    np.save(pred_rot_mat_cache_path, pred_rot_mat_cache)
    np.save(pred_kpts_cache_path, pred_kpts_cache)

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
    inference_yolo()