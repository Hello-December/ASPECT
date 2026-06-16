# ASPECT

ASPECT is a multi-stage monocular spacecraft pose-estimation pipeline for the
D4PED DFH-4 spacecraft dataset.

## Repository Structure

```text
ASPECT/
|-- model/
|   `-- refiner.py                         # Pose Transformer model
|-- scripts/
|   |-- train_yolo26s-pose_v1.py           # YOLO26s-Pose training
|   |-- train_yolo26m-pose_v1.py           # YOLO26m-Pose training
|   |-- preprocess_yolo_features.py        # YOLO feature extraction
|   |-- train_refiner_preprocessed_direct_load.py
|   |                                      # Pose Transformer training
|   |-- inference_yolo26.py                # YOLO26 + EPnP baseline
|   |-- inference_refiner.py               # Pose Transformer + EPnP inference
|   |-- inference_hybrid.py                # Hybrid ASPECT inference
|   `-- ukf/
|       |-- ukf_filter.py                  # Fixed-Q UKF backend
|       |-- ukf_filter_dyn_q.py            # NIS adaptive-Q UKF
|       |-- ukf_filter_sage_husa.py        # Sage-Husa adaptive UKF
|       `-- pose_precision_analyzer_v2.7.py
|                                          # Rotation-error analysis
|-- utils/
|   `-- tools.py
|-- draw_system_diagram.py
`-- draw_system_diagram_v2.py
```

Large generated files such as datasets, feature caches, checkpoints, and
training outputs should normally not be committed to GitHub. Keep directories
such as `files/`, `runs/`, `weights/`, `checkpoints/`, and large `*.pt` files
outside Git, or publish them separately through GitHub Releases, Kaggle, or
cloud storage.

## Environment Setup

The code is written in Python and uses PyTorch, Ultralytics YOLO, OpenCV, SciPy,
NumPy, Pandas, Matplotlib, TensorBoard, and tqdm. A CUDA-enabled GPU is strongly
recommended for training and inference.

Create a conda environment:

```bash
conda create -n aspect python=3.10 -y
conda activate aspect
```

Install PyTorch according to your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install ultralytics opencv-python numpy scipy pandas pyyaml tqdm matplotlib tensorboard openpyxl
```

If you use a different CUDA version, install the corresponding PyTorch build
from the official PyTorch installation selector.

## Dataset

The D4PED dataset is available on Kaggle:

https://www.kaggle.com/datasets/soranorinji/d4ped-a-dataset-for-dfh-4-pose-estimation

After downloading the dataset, place it in a local directory, for example:

```text
/path/to/D4PED/
```

The training scripts use Ultralytics-style dataset YAML files. Update the YAML
file path in the scripts before running them. The current local scripts contain
absolute paths such as:

```text
/home/chenxuyang/datasets/103_spacecraft/v2/...
```

Replace these paths with your own dataset location. A typical YAML file should
define the dataset root and the train/validation image folders, for example:

```yaml
path: /path/to/D4PED
train: train
val: val
test: test
names:
  0: spacecraft
```

Make sure the image files, YOLO pose labels, camera intrinsics, and 3D keypoint
definitions are consistent with the selected experiment. The inference scripts
also require the initial 3D spacecraft keypoints used by EPnP, loaded from a
`.mat` file in the current implementation.

## Important Configuration Notes

Before running the pipeline, check and edit the hard-coded paths and device IDs
inside the scripts:

- `sys.path.append(...)`: project root path
- dataset YAML path
- image sequence directory
- YOLO checkpoint path
- Pose Transformer checkpoint path
- output directory
- CUDA device ID
- camera intrinsic matrix
- 3D keypoint `.mat` path
- number of keypoints used by EPnP

The current codebase is experiment-oriented rather than fully CLI-configured.
For clean reproduction, edit these variables directly or refactor them into
command-line arguments.

## Training Pipeline

### 1. Train YOLO26-Pose

Edit `scripts/train_yolo26s-pose_v1.py`:

- `model = YOLO("yolo26s-pose.pt")`
- `data = ".../dongfanghong_4_coco.yaml"`
- `device`
- `batch`
- `imgsz`
- `project`
- `name`

Then run:

```bash
python scripts/train_yolo26s-pose_v1.py
```

For the medium model, use:

```bash
python scripts/train_yolo26m-pose_v1.py
```

The trained YOLO checkpoints are saved under the configured `project/name`
directory, usually in:

```text
runs/pose/models/D4PED/<experiment_name>/weights/best.pt
```

### 2. Preprocess YOLO Features

The Pose Transformer is trained on intermediate YOLO feature maps. Edit
`scripts/preprocess_yolo_features.py`:

- YOLO checkpoint path
- dataset YAML path
- dataset split: `train` or `val`
- feature output directory
- CUDA device

Then run:

```bash
python scripts/preprocess_yolo_features.py
```

Run this preprocessing for both training and validation splits. The default
script writes feature files to a directory such as:

```text
/tmp/d4ped_speedplus_features/train/
/tmp/d4ped_speedplus_features/val/
```

Each image produces one `.npy` feature file.

### 3. Train the Pose Transformer

Edit `scripts/train_refiner_preprocessed_direct_load.py`:

- `dataset_yaml_path`
- feature root, for example `/tmp/d4ped_speedplus_features/`
- `save_dir_root`
- `project_name`
- `device`
- `seq_length`
- `batch_size`

Then run:

```bash
python scripts/train_refiner_preprocessed_direct_load.py
```

The script saves periodic checkpoints, the best training checkpoint, the best
validation checkpoint, and TensorBoard logs under:

```text
files/refiner/<project_name>/
```

To monitor training:

```bash
tensorboard --logdir files/refiner/<project_name>/tensorboard
```

## Evaluation Pipeline

### 1. YOLO26 + EPnP Baseline

Edit `scripts/inference_yolo26.py`:

- `DEVICE`
- `WRITE_DIR`
- `img_dir`
- YOLO checkpoint path
- 3D keypoint `.mat` path
- camera intrinsic matrix
- EPnP keypoint count

Run:

```bash
python scripts/inference_yolo26.py
```

The script saves:

```text
pred_rot_euler_cache.npy
pred_rot_quat_cache.npy
pred_rot_mat_cache.npy
pred_kpts_cache.npy
pred_latency_cache.xlsx
```

### 2. Hybrid ASPECT Inference

Edit `scripts/inference_hybrid.py`:

- `WRITE_DIR`
- `SEQ_LEN`
- YOLO checkpoint path
- Pose Transformer checkpoint path
- image sequence directory
- 3D keypoint `.mat` path
- camera intrinsic matrix
- `kpts_num`
- CUDA device

Run:

```bash
python scripts/inference_hybrid.py
```

The hybrid policy first tries YOLO26 keypoints and EPnP. If the keypoint set is
incomplete or EPnP fails, it uses the cached YOLO feature window and the Pose
Transformer to generate replacement keypoints.

The output files have the same format as the YOLO baseline and can be passed to
the UKF and precision-analysis scripts.

### 3. UKF Backend

For the fixed-Q UKF backend, edit `scripts/ukf/ukf_filter.py`:

- `pred_poses_path`
- `gt_poses_path`
- inertia matrix if needed
- process noise `Q`
- observation noise `R`

Run:

```bash
python scripts/ukf/ukf_filter.py
```

The fixed-Q UKF output is saved under:

```text
files/ukf_result/normal/<experiment_name>/
```

Important outputs include:

```text
ukf_euler_angles_rad.npy
ukf_euler_angles_deg.npy
ukf_quaternions.npy
euler_gt_vs_ukf.png
euler_gt_vs_ukf_vs_raw.png
```

Adaptive UKF variants are available in:

```bash
python scripts/ukf/ukf_filter_dyn_q.py
python scripts/ukf/ukf_filter_sage_husa.py
```

### 4. Rotation-Error Analysis

Edit `scripts/ukf/pose_precision_analyzer_v2.7.py`:

- `euler_pose_prediction_path`
- `euler_pose_ground_truth_path`

Then run:

```bash
python scripts/ukf/pose_precision_analyzer_v2.7.py
```

The analyzer computes:

- per-axis Euler-angle errors
- quaternion geodesic rotation error in degrees
- mean Euler error
- mean quaternion rotation error

It writes:

```text
pose_precision_analysis.xlsx
pose_visualization.png
```

to the same directory as the prediction file.

## Typical End-to-End Workflow

```bash
# 1. Train YOLO26-Pose
python scripts/train_yolo26s-pose_v1.py

# 2. Extract YOLO features for train/val splits
python scripts/preprocess_yolo_features.py

# 3. Train Pose Transformer
python scripts/train_refiner_preprocessed_direct_load.py

# 4. Run hybrid inference
python scripts/inference_hybrid.py

# 5. Run fixed-Q UKF backend
python scripts/ukf/ukf_filter.py

# 6. Compute rotation errors
python scripts/ukf/pose_precision_analyzer_v2.7.py
```

## Outputs and Metrics

The main metric is attitude error, computed as the geodesic distance between
the estimated and ground-truth unit quaternions:

```text
theta_err = 2 * arccos(|q_est^T q_gt|)
```

The project also records inference latency and produces Euler-angle tracking
plots for qualitative analysis.

## Reproducibility Checklist

Before reporting results, record:

- dataset version and split
- YOLO checkpoint path
- Pose Transformer checkpoint path
- sequence length
- keypoint count used by EPnP
- camera intrinsic matrix
- 3D keypoint definition file
- UKF `Q` and `R` settings
- CUDA device and GPU model

## Citation

If you use this repository or the D4PED dataset in academic work, please cite
the corresponding ASPECT paper and the D4PED dataset page.

## License

License information has not been specified yet. Add a `LICENSE` file before
publishing the repository.
