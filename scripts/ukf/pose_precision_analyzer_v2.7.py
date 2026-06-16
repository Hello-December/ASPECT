import numpy as np
import os
import pandas as pd
import math
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

''' begin editable params '''
# euler pose prediction path (must in radian)
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_hybrid_v1.1_train_3_random_light_0.1_40_0.08_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"

# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_hybrid_v1.1_train_3_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_hybrid_v1.1_train_3_random_light_0.1_0.7_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_hybrid_v1.1_train_3_random_light_0.1_10_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_hybrid_v1.1_train_3_random_light_0.1_40_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_hybrid_v1.1_train_3_random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"

# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_refiner_v1.1_train_3_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_refiner_v1.1_train_3_random_light_0.1_0.7_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_refiner_v1.1_train_3_random_light_0.1_10_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_refiner_v1.1_train_3_random_light_0.1_40_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_refiner_v1.1_train_3_random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"

# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_yolo26s_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_yolo26s_random_light_0.1_0.7_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_yolo26s_random_light_0.1_10_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_yolo26s_random_light_0.1_40_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"
euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/ukf_result/sage_husa/test_yolo26s_random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12/ukf_euler_angles_rad.npy"

# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_refiner_v1.1_train_3_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_refiner_v1.1_train_3_random_light_0.1_40_0.08_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"

# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_yolo26s_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_yolo26s_random_light_0.1_10_0.08_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"
# euler_pose_prediction_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_yolo26s_random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"

# euler pose ground truth path (must in radian)
# euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/0.0_0.7_0.0_0.1_1000_v2.npy"
# euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/0.1_0.7_0.08_0.1_1000_v2.npy"
# euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/0.1_10_0.08_0.1_1000_v2.npy"
# euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/0.1_40_0.08_0.1_1000_v2.npy"
euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/0.1_60_0.08_0.1_1000_v2.npy"

# euler_pose_ground_truth_path = "/home/chenxuyang/PythonProjects/ASPECT/files/predefined_poses/speedplusv2/speedplusv2_synthetic_validation_pose.npy"
''' end editable params '''

# define result write directory
result_write_dir = os.path.dirname(euler_pose_prediction_path)
result_write_name = "pose_precision_analysis.xlsx" 
result_write_path = os.path.join(result_write_dir, result_write_name)
result_image_name = "pose_visualization.png"
result_image_path = os.path.join(result_write_dir, result_image_name)
if os.path.exists(result_write_path):
    print(f"Warning: \n---------------\n{result_write_path}\n---------------\nalready exists, delete it and continue? (y/n)")
    print(">> ", end="")
    if input() == "y":
        os.remove(result_write_path)
    else:
        exit()

# load euler pose prediction
euler_pose_prediction = np.load(euler_pose_prediction_path)

# load euler pose ground truth
euler_pose_ground_truth = np.load(euler_pose_ground_truth_path)

# length of prediction
length_of_prediction = euler_pose_prediction.shape[0]

# define a list to store the error of each prediction
error_of_each_prediction = []
error_of_each_prediction_quat_degree = []

for i in range(length_of_prediction):
    this_pred_euler = np.degrees(euler_pose_prediction[i])
    this_gt_euler = np.degrees(euler_pose_ground_truth[i])
    this_delta_euler = np.abs(this_pred_euler - this_gt_euler)
    # this_delta_euler = this_pred_euler - this_gt_euler
    for j in range(3):
        if this_delta_euler[j] > 180:
            this_delta_euler[j] = this_delta_euler[j] - 360 # make sure the error is in [0, 180]
        if this_delta_euler[j] <= -180:
            this_delta_euler[j] = this_delta_euler[j] + 360 # make sure the error is in [0, 180]
    error_of_each_prediction.append(this_delta_euler)

    this_pred_quat = R.from_euler('xyz', this_pred_euler, degrees=True).as_quat(scalar_first=True)
    this_gt_quat = R.from_euler('xyz', this_gt_euler, degrees=True).as_quat(scalar_first=True)
    this_delta_quat_dot = np.abs(np.dot(this_pred_quat, this_gt_quat))
    this_delta_quat_dot = np.minimum(this_delta_quat_dot, 1.0)
    this_delta_quat_degree = np.degrees(np.arccos(this_delta_quat_dot) * 2)
    error_of_each_prediction_quat_degree.append(this_delta_quat_degree)

# convert error_of_each_prediction to numpy array
error_of_each_prediction = np.array(error_of_each_prediction)
error_of_each_prediction_quat_degree = np.array(error_of_each_prediction_quat_degree)

# calculate the mean error
mean_error = np.mean(error_of_each_prediction)
print(f"Mean error: {mean_error} degrees in euler")
mean_error_quat_degree = np.mean(error_of_each_prediction_quat_degree)
print(f"Mean error: {mean_error_quat_degree} degrees in quat")

# create a dataframe to store the error of each prediction
df = pd.DataFrame(error_of_each_prediction, columns=["delta_x", "delta_y", "delta_z"])
# create a dataframe to store the error of each prediction
df_quat_degree = pd.DataFrame(error_of_each_prediction_quat_degree, columns=["degree"])

# create a dataframe to store the stats
stats_df = pd.DataFrame({
    "mean_error": [mean_error],
    "mean_error_quat_degree": [mean_error_quat_degree]
})

# save the two dataframes to the same excel file in different sheets
with pd.ExcelWriter(result_write_path) as writer:
    df.to_excel(writer, sheet_name='xyz欧拉角误差', index=False)
    df_quat_degree.to_excel(writer, sheet_name='四元数误差', index=False)
    stats_df.to_excel(writer, sheet_name='统计结果', index=False)

# plot x y z sequence in seperate plots
euler_pose_prediction = np.degrees(euler_pose_prediction)
euler_pose_ground_truth = np.degrees(euler_pose_ground_truth)
for i in range(len(euler_pose_prediction)):
    for j in range(3):
        if abs(euler_pose_prediction[i, j] - euler_pose_ground_truth[i, j]) // 180 > 0:
            euler_pose_prediction[i, j] += 360 * math.ceil(abs(euler_pose_prediction[i, j] - euler_pose_ground_truth[i, j]) / 360) * np.sign(-euler_pose_prediction[i, j] + euler_pose_ground_truth[i, j])
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(euler_pose_prediction[:, 0], label="prediction")
plt.plot(euler_pose_ground_truth[:, 0][:length_of_prediction], label="ground truth")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(euler_pose_prediction[:, 1], label="prediction")
plt.plot(euler_pose_ground_truth[:, 1][:length_of_prediction], label="ground truth")
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(euler_pose_prediction[:, 2], label="prediction")
plt.plot(euler_pose_ground_truth[:, 2][:length_of_prediction], label="ground truth")
plt.legend()

plt.tight_layout()
plt.savefig(result_image_path)
plt.close()
