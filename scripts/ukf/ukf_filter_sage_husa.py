# for VSCode
import sys
sys.path.append("/home/chenxuyang/PythonProjects/ASPECT/")

import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def default_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    """Weighted mean for non-angular states."""
    return (Wm[:, None] * sigmas).sum(axis=0)


def mean_with_angles(sigmas: np.ndarray, Wm: np.ndarray, angle_idx=()) -> np.ndarray:
    """
    Weighted mean with circular mean for specified angle indices.
    sigmas: (2n+1, dim)
    Wm:     (2n+1,)
    """
    x = default_mean(sigmas, Wm).copy()
    for k in angle_idx:
        s = (Wm * np.sin(sigmas[:, k])).sum()
        c = (Wm * np.cos(sigmas[:, k])).sum()
        x[k] = np.arctan2(s, c)
    return x


def residual_with_angles(a: np.ndarray, b: np.ndarray, angle_idx=()) -> np.ndarray:
    """Residual (a-b) with wrapping on angle indices."""
    y = a - b
    for k in angle_idx:
        y[k] = wrap_angle(y[k])
    return y


class UKF:
    """
    Additive-noise UKF (Scaled sigma points / Julier-Uhlmann-Merwe style).

    x_{k+1} = f(x_k, *fx_args) + w,    w ~ N(0, Q)
    z_k     = h(x_k, *hx_args) + v,    v ~ N(0, R)
    """

    def __init__(
        self,
        dim_x: int,  # 
        dim_z: int,  # 
        fx,  # 运动学公式  # D4PED 7th equation
        hx,  # 测量公式 (np.ones(1, 4))
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        x0: np.ndarray | None = None,  # x0初值 默认0 
        P0: np.ndarray | None = None, # P0初值 默认单位矩阵 dim(7, 7)
        x_angle_idx=(),
        z_angle_idx=(),
    ):
        self.n = int(dim_x)
        self.m = int(dim_z)

        self.fx = fx
        self.hx = hx

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float).reshape(-1)
        self.P = np.eye(self.n) if P0 is None else np.array(P0, dtype=float)

        # UT parameters
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.lmbda = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lmbda)

        # weights
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lmbda)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lmbda / (self.n + self.lmbda)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        # angle handling
        self.x_angle_idx = tuple(x_angle_idx)
        self.z_angle_idx = tuple(z_angle_idx)

        # caches
        self._sigmas_f = None  # predicted sigmas in state space

        # -------------------------
        # Sage-Husa Adaptive Q estimation
        # Directly estimates Q from innovation sequence,
        # no need for a good Q0 initial value.
        # -------------------------
        self.adapt_Q = True
        self.sage_husa_b = 0.92      # 遗忘因子 b ∈ (0,1), 越小记忆越短, 建议0.90~0.95
        self._sh_step = 0            # 步数计数器
        self._P_prior = None         # predict后的先验P，供update使用
        self.Q_floor = 1e-10          # Q对角线下限，防止Q塌缩到0，增加数值稳定性



    # legacy
    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points around x with covariance P."""
        # 增强数值稳定性：确保矩阵对称且半正定
        P = 0.5 * (P + P.T)  # symmetrize
        
        # 添加更强的正则化以确保矩阵正定
        jitter = 1e-6  # 增加 jitter 值
        max_jitter = 1e-3  # 最大 jitter 值
        
        # 尝试 Cholesky 分解，逐步增加 jitter
        for attempt in range(10):
            current_jitter = jitter * (10 ** attempt)
            if current_jitter > max_jitter:
                current_jitter = max_jitter
            
            try:
                S = np.linalg.cholesky(P + current_jitter * np.eye(self.n))
                break
            except np.linalg.LinAlgError:
                if attempt == 9:  # 最后一次尝试
                    # 如果所有尝试都失败，使用特征值分解强制正定
                    eigvals, eigvecs = np.linalg.eigh(P)
                    eigvals = np.maximum(eigvals, 1e-8)  # 设置最小特征值
                    P = eigvecs @ np.diag(eigvals) @ eigvecs.T
                    S = np.linalg.cholesky(P + max_jitter * np.eye(self.n))
                continue

        sigmas = np.empty((2 * self.n + 1, self.n), dtype=float)
        sigmas[0] = x
        for i in range(self.n):
            d = self.gamma * S[:, i]
            sigmas[i + 1] = x + d
            sigmas[self.n + i + 1] = x - d
        return sigmas

    def _x_mean(self, sigmas: np.ndarray) -> np.ndarray:
        return mean_with_angles(sigmas, self.Wm, self.x_angle_idx)

    def _z_mean(self, sigmas: np.ndarray) -> np.ndarray:
        return mean_with_angles(sigmas, self.Wm, self.z_angle_idx)

    def predict(self, fx_args=()):
        """Time update."""
        sigmas = self._sigma_points(self.x, self.P)

        sigmas_f = np.empty_like(sigmas)
        for i, s in enumerate(sigmas):
            sigmas_f[i] = np.asarray(self.fx(s, *fx_args), dtype=float).reshape(-1)

        x_pred = self._x_mean(sigmas_f)

        P_pred = np.zeros((self.n, self.n), dtype=float)
        for i in range(sigmas_f.shape[0]):
            dx = residual_with_angles(sigmas_f[i], x_pred, self.x_angle_idx)
            P_pred += self.Wc[i] * np.outer(dx, dx)
        # P_pred += self.Q
        # use adaptive Q (Sage-Husa: self.Q is updated each step)
        P_pred += self.Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred
        self.P = P_pred
        self._P_prior = P_pred.copy()  # save for Sage-Husa update
        self._sigmas_f = sigmas_f  # cache for update
        return self.x, self.P

    def update(self, z: np.ndarray, hx_args=()):
        """Measurement update."""
        if self._sigmas_f is None:
            # if user calls update without predict, use sigma points from current posterior
            self._sigmas_f = self._sigma_points(self.x, self.P)

        z = np.asarray(z, dtype=float).reshape(-1)

        # transform sigma points into measurement space
        sigmas_z = np.empty((2 * self.n + 1, self.m), dtype=float)
        for i, s in enumerate(self._sigmas_f):
            sigmas_z[i] = np.asarray(self.hx(s, *hx_args), dtype=float).reshape(-1)

        z_pred = self._z_mean(sigmas_z)

        # innovation covariance
        Pzz = np.zeros((self.m, self.m), dtype=float)
        for i in range(sigmas_z.shape[0]):
            dz = residual_with_angles(sigmas_z[i], z_pred, self.z_angle_idx)
            Pzz += self.Wc[i] * np.outer(dz, dz)
        Pzz += self.R
        Pzz = 0.5 * (Pzz + Pzz.T)

        # cross covariance
        Pxz = np.zeros((self.n, self.m), dtype=float)
        for i in range(sigmas_z.shape[0]):
            dx = residual_with_angles(self._sigmas_f[i], self.x, self.x_angle_idx)
            dz = residual_with_angles(sigmas_z[i], z_pred, self.z_angle_idx)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        # Use solve instead of inv for stability
        K = Pxz @ np.linalg.solve(Pzz, np.eye(self.m))

        y = residual_with_angles(z, z_pred, self.z_angle_idx)  # innovation

        self.x = self.x + K @ y
        self.P = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # --- Sage-Husa adaptive Q estimation ---
        if getattr(self, "adapt_Q", False) and self._P_prior is not None:
            self._sh_step += 1
            
            # 自适应遗忘因子：初期学习快，后期稳定
            if self._sh_step < 100:  # 前100步快速学习
                b = 0.85
            else:
                b = self.sage_husa_b
                
            d_k = (1.0 - b) / (1.0 - b ** (self._sh_step + 1))  # 遗忘加权因子

            # Q_sample = K * y * y^T * K^T + P_post - P_prior
            Q_sample = K @ np.outer(y, y) @ K.T + self.P - self._P_prior
            Q_sample = 0.5 * (Q_sample + Q_sample.T)

            # 指数加权递推
            self.Q = (1.0 - d_k) * self.Q + d_k * Q_sample
            self.Q = 0.5 * (self.Q + self.Q.T)

            # 确保Q半正定：将负特征值截断为floor
            eigvals, eigvecs = np.linalg.eigh(self.Q)
            eigvals = np.maximum(eigvals, self.Q_floor)
            self.Q = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # 限制Q矩阵的最大值，防止过度放大
            max_Q = 1e-3  # Q矩阵元素最大值
            self.Q = np.clip(self.Q, -max_Q, max_Q)

        # 增强协方差矩阵P的数值稳定性
        # 确保P矩阵对称且半正定
        self.P = 0.5 * (self.P + self.P.T)  # 强制对称
        
        # 检查P矩阵的特征值，确保正定性
        eigvals, eigvecs = np.linalg.eigh(self.P)
        if np.any(eigvals < 1e-10):  # 如果存在很小的特征值
            # 将负特征值或过小的特征值设为最小值
            eigvals = np.maximum(eigvals, 1e-8)
            self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # clear cache
        self._sigmas_f = None
        return self.x, self.P, y, K

def build_I(Ixx, Iyy, Izz, Ixy, Iyz, Ixz):
    """
    Build inertia matrix from moment of inertia components.
    
    Args:
        Ixx, Iyy, Izz: Moment of inertia about x, y, z axes
        Ixy, Iyz, Ixz: Cross moment of inertia components
    
    Returns:
        3x3 inertia matrix
    """
    return np.array([
        [Ixx, -Ixy, -Ixz],
        [-Ixy, Iyy, -Iyz],
        [-Ixz, -Iyz, Izz],
    ])


def ukf_infer():
    # 替换所有 \ 为 /，支持跨平台，无转义错误
    pred_poses_path = r"./files/rot_speed_analysis/test_yolo26s_random_light_0.1_60_0.08_0.1_1000_v2_2000imgs_v12/pred_rot_euler_cache.npy"
    gt_poses_path = r"./files/predefined_poses/0.1_60_0.08_0.1_1000_v2.npy"

    pred_poses = np.load(pred_poses_path)
    gt_poses = np.load(gt_poses_path)

    # 再保险：长度对齐
    T_all = min(pred_poses.shape[0], gt_poses.shape[0])
    pred_poses = pred_poses[:T_all]
    gt_poses   = gt_poses[:T_all]

    print("pred_poses:", pred_poses.shape, "gt_poses:", gt_poses.shape, "T_all:", T_all)

    # Inertia matrix
    Ixx = 20161.3
    Iyy = 4619.382
    Izz = 18216.06
    Ixy = -58.045
    Iyz = -149.798
    Ixz = -65.455
    my_I = build_I(Ixx, Iyy, Izz, Ixy, Iyz, Ixz)

    pg_bar = tqdm.tqdm(total=pred_poses.shape[0])

    def state_space_representation(x_k, I=my_I):
        """
        State space representation for spacecraft attitude dynamics and kinematics.
        
        Args:
            x_k: 7-dimensional state vector [q0, q1, q2, q3, wx, wy, wz]
            I: 3x3 inertia matrix (default: identity matrix)
        
        Returns:
            7-dimensional state derivative vector [dq0, dq1, dq2, dq3, dwx, dwy, dwz]
        """
        # Extract quaternion and angular velocity from state vector
        q = x_k[:4]
        omega = x_k[4:]
        
        # Compute quaternion derivative using Omega matrix
        Omega = np.array([
            [0, -omega[0], -omega[1], -omega[2]],
            [omega[0], 0, omega[2], -omega[1]],
            [omega[1], -omega[2], 0, omega[0]],
            [omega[2], omega[1], -omega[0], 0]
        ])
        q_dot = 0.5 * Omega @ q
        
        # Compute angular velocity derivative using Euler's equation for free rotation
        H = I @ omega
        omega_cross_H = np.cross(omega, H)
        omega_dot = -np.linalg.inv(I) @ omega_cross_H
        
        # Combine derivatives into 7-dimensional state derivative vector
        x_dot = np.concatenate([q_dot, omega_dot])
        return x_dot

    def enforce_quat_hemisphere(q: np.ndarray) -> np.ndarray:
        """
        Enforce quaternion to lie in a consistent hemisphere to reduce sign flips.
        Here we force scalar part q0 >= 0.
        """
        if q[0] < 0:
            return -q
        return q

    def rk4_step(x: np.ndarray, dt=0.1, I=my_I) -> np.ndarray:
        k1 = state_space_representation(x, I)
        k2 = state_space_representation(x + 0.5 * dt * k1, I)
        k3 = state_space_representation(x + 0.5 * dt * k2, I)
        k4 = state_space_representation(x + dt * k3, I)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Normalize quaternion and enforce hemisphere
        # x_next[:4] = enforce_quat_hemisphere(quat_normalize(x_next[:4]))
        return x_next

    def quat_normalize(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n <= 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    my_ukf = UKF(
        dim_x=7,
        dim_z=4,
        fx=rk4_step,
        hx=lambda x: x[:4],
        # Q=np.array([
        #     [1e-8, 0, 0, 0, 0, 0, 0],
        #     [0, 1e-5, 0, 0, 0, 0, 0],
        #     [0, 0, 1e-4, 0, 0, 0, 0],
        #     [0, 0, 0, 1e-6, 0, 0, 0],
        #     [0, 0, 0, 0, 1e-3, 0, 0],
        #     [0, 0, 0, 0, 0, 1e-3, 0],
        #     [0, 0, 0, 0, 0, 0, 1e-3],
        # ]),
        Q=np.eye(7) * 1,
        R=np.eye(4) * 1e-3,
        # R=np.array([
        #     [1e-5, 0, 0, 0],
        #     [0, 1e-3, 0, 0],
        #     [0, 0, 1e-5, 0],
        #     [0, 0, 0, 1e-4]
        # ]),
        x_angle_idx=(),
        z_angle_idx=(),
        x0 = np.zeros(7),
        P0 = np.eye(7) * 1e-3,
    )

    # Data collection for visualization
    ukf_quaternions = []
    ukf_angular_velocities = []
    gt_quaternions = []
    rk4_values = []
    input_quaternions = []
    q_scales = []  # now tracks trace(Q) for visualization


    for i in range(pred_poses.shape[0]):
        pred_euler_rad = pred_poses[i]
        gt_euler_rad = gt_poses[i]

        pred_quat = R.from_euler("xyz", pred_euler_rad, degrees=False).as_quat()
        gt_quat = R.from_euler("xyz", gt_euler_rad, degrees=False).as_quat()

        # Store input quaternion
        input_quaternions.append(pred_quat.copy())
        
        # Store current state before prediction (for rk4 input)
        current_state = my_ukf.x.copy()
        
        # Run predict to get rk4 result
        my_ukf.predict()
        
        # Store rk4 returned value (predicted state)
        rk4_values.append(my_ukf.x.copy())
        
        # Update with measurement
        my_ukf.update(z=pred_quat)
        
        # Store UKF predicted results after update
        ukf_quaternions.append(my_ukf.x[:4].copy())
        ukf_angular_velocities.append(my_ukf.x[4:].copy())
        gt_quaternions.append(gt_quat.copy())
        q_scales.append(np.trace(my_ukf.Q))


        pg_bar.update(1)
        pg_bar.set_postfix({
            "q0": f"{my_ukf.x[0]:.2f}",
            "q1": f"{my_ukf.x[1]:.2f}",
            "q2": f"{my_ukf.x[2]:.2f}",
            "q3": f"{my_ukf.x[3]:.2f}",
            "wx": f"{my_ukf.x[4]:.2f}",
            "wy": f"{my_ukf.x[5]:.2f}",
            "wz": f"{my_ukf.x[6]:.2f}",
        })

    pg_bar.close()

    # Convert lists to numpy arrays
    ukf_quaternions = np.array(ukf_quaternions)
    ukf_angular_velocities = np.array(ukf_angular_velocities)
    gt_quaternions = np.array(gt_quaternions)
    rk4_values = np.array(rk4_values)
    input_quaternions = np.array(input_quaternions)
    q_scales = np.array(q_scales)


    subdirectory_name = os.path.split(os.path.dirname(pred_poses_path))[-1]
    
    # Create output directory
    project_root = "./"
    output_dir = os.path.join(project_root, "files", "ukf_result", "sage_husa", subdirectory_name)
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 6 plots total:
    #   Quaternion: (GT vs UKF), (Meas vs UKF), (RK4 vs UKF)
    #   Euler(deg): (GT vs UKF), (Meas vs UKF), (RK4 vs UKF)
    # NOTE:
    #   - input Euler and GT Euler already exist as pred_poses/gt_poses (rad)
    #   - UKF/RK4 Euler are converted from quaternions (sanitized)
    # ============================================================

    T = ukf_quaternions.shape[0]
    time_steps = np.arange(T)

    # data sources (truncate to T for safety)
    meas_quats = input_quaternions[:T]
    gt_quats   = gt_quaternions[:T]
    ukf_quats  = ukf_quaternions[:T]
    rk4_quats  = rk4_values[:T, :4]

    # input/gt euler already have (rad) -> convert to deg and truncate
    input_euler_angles = np.degrees(pred_poses[:T])
    gt_euler_angles    = np.degrees(gt_poses[:T])

    # Quaternion sanitize for SciPy conversion (avoid zero-norm crash)
    def sanitize_quat_for_scipy(Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        Q = np.asarray(Q, dtype=float).copy()
        norms = np.linalg.norm(Q, axis=1)
        bad = norms < eps

        if np.any(bad):
            last_good = None
            for k in range(Q.shape[0]):
                if norms[k] >= eps:
                    last_good = Q[k].copy()
                else:
                    if last_good is not None:
                        Q[k] = last_good
                    else:
                        Q[k] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)  # xyzw identity

        norms2 = np.linalg.norm(Q, axis=1, keepdims=True)
        Q = Q / np.maximum(norms2, eps)
        return Q

    ukf_quats_s = sanitize_quat_for_scipy(ukf_quats)
    rk4_quats_s = sanitize_quat_for_scipy(rk4_quats)

    # Convert UKF/RK4 quaternion -> euler(deg)
    ukf_euler_angles = np.array([R.from_quat(q).as_euler('xyz', degrees=True) for q in ukf_quats_s])
    rk4_euler_angles = np.array([R.from_quat(q).as_euler('xyz', degrees=True) for q in rk4_quats_s])
    
    # Convert UKF quaternion -> euler(rad)
    ukf_euler_angles_rad = np.array([R.from_quat(q).as_euler('xyz', degrees=False) for q in ukf_quats_s])

    quaternion_labels = ['q0', 'q1', 'q2', 'q3']
    euler_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']

    # -------------------------
    # Helpers to plot comparisons
    # -------------------------
    def plot_quat_compare(a_quats, b_quats, a_name, b_name, filename):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time_steps, a_quats[:, i], label=f'{a_name} {quaternion_labels[i]}',
                    linewidth=2, linestyle='--')
            ax.plot(time_steps, b_quats[:, i], label=f'{b_name} {quaternion_labels[i]}',
                    linewidth=2)

            ax.set_ylabel(quaternion_labels[i])
            ax.set_title(f'Quaternion {quaternion_labels[i]}: {a_name} vs {b_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_euler_compare(a_euler, b_euler, a_name, b_name, filename):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time_steps, a_euler[:, i], label=f'{a_name} {euler_labels[i]}',
                    linewidth=2, linestyle='--')
            ax.plot(time_steps, b_euler[:, i], label=f'{b_name} {euler_labels[i]}',
                    linewidth=2)

            ax.set_ylabel(euler_labels[i])
            ax.set_title(f'Euler {euler_labels[i]}: {a_name} vs {b_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_quat_compare_three(a_quats, b_quats, c_quats, a_name, b_name, c_name, filename):
        """Plot three-way comparison for quaternions."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time_steps, a_quats[:, i], label=f'{a_name} {quaternion_labels[i]}',
                    linewidth=2, linestyle='--', color='red')
            ax.plot(time_steps, b_quats[:, i], label=f'{b_name} {quaternion_labels[i]}',
                    linewidth=2, color='blue')
            ax.plot(time_steps, c_quats[:, i], label=f'{c_name} {quaternion_labels[i]}',
                    linewidth=2, linestyle=':', color='green')

            ax.set_ylabel(quaternion_labels[i])
            ax.set_title(f'Quaternion {quaternion_labels[i]}: {a_name} vs {b_name} vs {c_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    def plot_euler_compare_three(a_euler, b_euler, c_euler, a_name, b_name, c_name, filename):
        """Plot three-way comparison for Euler angles."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(time_steps, a_euler[:, i], label=f'{a_name} {euler_labels[i]}',
                    linewidth=2, linestyle='--', color='red')
            ax.plot(time_steps, b_euler[:, i], label=f'{b_name} {euler_labels[i]}',
                    linewidth=2, color='blue')
            ax.plot(time_steps, c_euler[:, i], label=f'{c_name} {euler_labels[i]}',
                    linewidth=2, linestyle=':', color='green')

            ax.set_ylabel(euler_labels[i])
            ax.set_title(f'Euler {euler_labels[i]}: {a_name} vs {b_name} vs {c_name}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        axes[-1].set_xlabel('Time Step')
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        return path

    # -------------------------
    # 1) GT vs UKF
    # -------------------------
    quat_gt_ukf_path = plot_quat_compare(gt_quats, ukf_quats, 'GT', 'UKF',
                                        'quat_gt_vs_ukf.png')
    euler_gt_ukf_path = plot_euler_compare(gt_euler_angles, ukf_euler_angles, 'GT', 'UKF',
                                          'euler_gt_vs_ukf.png')

    # -------------------------
    # 2) Measurement(Input) vs UKF
    # -------------------------
    quat_meas_ukf_path = plot_quat_compare(meas_quats, ukf_quats, 'Meas', 'UKF',
                                          'quat_meas_vs_ukf.png')
    euler_meas_ukf_path = plot_euler_compare(input_euler_angles, ukf_euler_angles, 'Meas', 'UKF',
                                            'euler_meas_vs_ukf.png')

    # -------------------------
    # 3) RK4 vs UKF
    # -------------------------
    quat_rk4_ukf_path = plot_quat_compare(rk4_quats, ukf_quats, 'RK4', 'UKF',
                                         'quat_rk4_vs_ukf.png')
    euler_rk4_ukf_path = plot_euler_compare(rk4_euler_angles, ukf_euler_angles, 'RK4', 'UKF',
                                           'euler_rk4_vs_ukf.png')

    # -------------------------
    # 4) GT vs UKF vs Raw (三向对比)
    # -------------------------
    quat_gt_ukf_raw_path = plot_quat_compare_three(gt_quats, ukf_quats, meas_quats, 'GT', 'UKF', 'Raw',
                                                  'quat_gt_vs_ukf_vs_raw.png')
    euler_gt_ukf_raw_path = plot_euler_compare_three(gt_euler_angles, ukf_euler_angles, input_euler_angles, 'GT', 'UKF', 'Raw',
                                                    'euler_gt_vs_ukf_vs_raw.png')

    # -------------------------
    # Plot Q_scale over time
    # -------------------------
    T_plot = len(q_scales)
    time_steps_q = np.arange(T_plot)

    plt.figure(figsize=(12, 4))
    plt.plot(time_steps_q, q_scales, linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("trace(Q)")
    plt.title("Adaptive Q trace (Sage-Husa) over time")
    plt.grid(True, alpha=0.3)

    qscale_plot_path = os.path.join(output_dir, "q_scale.png")
    plt.tight_layout()
    plt.savefig(qscale_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # also save npy for later analysis
    np.save(os.path.join(output_dir, "q_scale.npy"), q_scales)

    print(f"  - Q trace plot: {qscale_plot_path}")




    print(f"UKF prediction results saved to: {output_dir}")
    print(f"  - Quaternion GT vs UKF:   {quat_gt_ukf_path}")
    print(f"  - Euler GT vs UKF:        {euler_gt_ukf_path}")
    print(f"  - Quaternion Meas vs UKF: {quat_meas_ukf_path}")
    print(f"  - Euler Meas vs UKF:      {euler_meas_ukf_path}")
    print(f"  - Quaternion RK4 vs UKF:  {quat_rk4_ukf_path}")
    print(f"  - Euler RK4 vs UKF:       {euler_rk4_ukf_path}")
    print(f"  - Quaternion GT vs UKF vs Raw: {quat_gt_ukf_raw_path}")
    print(f"  - Euler GT vs UKF vs Raw:      {euler_gt_ukf_raw_path}")

    # -------------------------
    # Save .npy arrays (like before)
    # -------------------------
    np.save(os.path.join(output_dir, 'ukf_quaternions.npy'), ukf_quats)
    np.save(os.path.join(output_dir, 'gt_quaternions.npy'), gt_quats)
    # np.save(os.path.join(output_dir, 'meas_quaternions.npy'), meas_quats)
    # np.save(os.path.join(output_dir, 'rk4_quaternions.npy'), rk4_quats)

    np.save(os.path.join(output_dir, 'ukf_euler_angles_deg.npy'), ukf_euler_angles)
    np.save(os.path.join(output_dir, 'gt_euler_angles_deg.npy'), gt_euler_angles)
    # np.save(os.path.join(output_dir, 'meas_euler_angles_deg.npy'), input_euler_angles)
    # np.save(os.path.join(output_dir, 'rk4_euler_angles_deg.npy'), rk4_euler_angles)
    
    # Save UKF Euler angles in radians
    np.save(os.path.join(output_dir, 'ukf_euler_angles_rad.npy'), ukf_euler_angles_rad)

    # # (optional) also save sanitized quaternions used for Euler conversion
    # np.save(os.path.join(output_dir, 'ukf_quaternions_sanitized.npy'), ukf_quats_s)
    # np.save(os.path.join(output_dir, 'rk4_quaternions_sanitized.npy'), rk4_quats_s)

    print("  - Saved npy files: ukf/gt/meas/rk4 quaternions + euler(deg) + euler(rad) (+ optional sanitized)")

if __name__ == "__main__":
    ukf_infer()
