import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

# ==================== 可编辑参数 ====================
xlsx_path = "/home/chenxuyang/PythonProjects/ASPECT/files/rot_speed_analysis/test_hybrid_v1.1_latency_train_3_random_light_0.0_0.7_0.0_0.1_1000_v2_2000imgs_v12/pred_latency_cache.xlsx"
save_dir = os.path.dirname(xlsx_path)
warmup_frames = 1  # 跳过前 N 帧（CUDA 预热），设为 0 则包含所有帧
outlier_upper_percentile = 99.5  # 纵轴上限基于此百分位值计算
outlier_margin = 1.3  # 上限 = P{outlier_upper_percentile} * outlier_margin
# ==================================================

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_prop = fm.FontProperties(fname=font_path)
font_prop_bold = fm.FontProperties(fname=font_path, weight="bold")
plt.rcParams["font.family"] = font_prop.get_name()
fm.fontManager.addfont(font_path)
plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_excel(xlsx_path)
frame_indices = df["frame_idx"].values
latency_ms = df["latency_ms"].values
total_frames = len(latency_ms)

if warmup_frames > 0:
    plot_indices = frame_indices[warmup_frames:]
    plot_latency = latency_ms[warmup_frames:]
    warmup_info = f"（已跳过前 {warmup_frames} 帧 CUDA 预热）"
else:
    plot_indices = frame_indices
    plot_latency = latency_ms
    warmup_info = ""

avg_latency = np.mean(plot_latency)

ylim_upper = np.percentile(plot_latency, outlier_upper_percentile) * outlier_margin
n_clipped = np.sum(plot_latency > ylim_upper)
clipped_info = f"\n（{n_clipped} 帧超出显示范围，已标注）" if n_clipped > 0 else ""

# ==================== 折线图 ====================
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle(f"每帧推理延迟 {warmup_info}".strip(), fontproperties=font_prop_bold, fontsize=16, y=0.97)

ax.plot(plot_indices, plot_latency, color="#2196F3", linewidth=0.8, alpha=0.85, label="每帧延迟")
ax.axhline(y=avg_latency, color="#F44336", linestyle="--", linewidth=1.5, label=f"平均延迟 ({avg_latency:.2f} ms)")
ax.set_xlabel("帧索引", fontproperties=font_prop, fontsize=13)
ax.set_ylabel("延迟 (ms)", fontproperties=font_prop, fontsize=13)
ax.set_xlim(plot_indices[0], plot_indices[-1])
ax.set_ylim(0, ylim_upper)
ax.legend(prop=font_prop, fontsize=12)
ax.grid(True, alpha=0.3)

if n_clipped > 0:
    clipped_mask = plot_latency > ylim_upper
    ax.scatter(plot_indices[clipped_mask], np.full(n_clipped, ylim_upper * 0.97),
               marker="^", color="#F44336", s=30, alpha=0.8, label=f"超出显示范围 ({n_clipped} 帧)")
    ax.legend(prop=font_prop, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.94])
line_save_path = os.path.join(save_dir, "latency_line.png")
fig.savefig(line_save_path, dpi=200, bbox_inches="tight")
print(f"折线图已保存: {line_save_path}")
plt.close(fig)

# ==================== 直方图 ====================
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle(f"推理延迟分布直方图 {warmup_info}".strip(), fontproperties=font_prop_bold, fontsize=16, y=0.97)

n_bins = max(50, int(ylim_upper / 2))
ax.hist(plot_latency[plot_latency <= ylim_upper], bins=n_bins, range=(0, ylim_upper),
        color="#4CAF50", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(x=avg_latency, color="#F44336", linestyle="--", linewidth=1.5, label=f"平均 {avg_latency:.2f} ms")
ax.set_xlabel("延迟 (ms)", fontproperties=font_prop, fontsize=13)
ax.set_ylabel("帧数", fontproperties=font_prop, fontsize=13)
ax.set_xlim(0, ylim_upper)
ax.legend(prop=font_prop, fontsize=12)
ax.grid(True, alpha=0.3)

if n_clipped > 0:
    ax.text(0.98, 0.95, f"{n_clipped} 帧超出显示范围 (>{ylim_upper:.0f} ms)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color="#F44336", fontproperties=font_prop,
            bbox=dict(facecolor="white", edgecolor="#F44336", alpha=0.8, boxstyle="round,pad=0.4"))

plt.tight_layout(rect=[0, 0, 1, 0.94])
hist_save_path = os.path.join(save_dir, "latency_histogram.png")
fig.savefig(hist_save_path, dpi=200, bbox_inches="tight")
print(f"直方图已保存: {hist_save_path}")
plt.close(fig)

# ==================== 统计信息 ====================
percentiles = np.percentile(plot_latency, [1, 5, 10, 25, 50, 75, 90, 95, 99])
print(f"\n延迟统计 (ms){warmup_info}:")
print(f"  总帧数           : {total_frames}")
print(f"  统计帧数         : {len(plot_latency)}（跳过前 {warmup_frames} 帧）")
print(f"  超出显示范围帧数 : {n_clipped}")
print(f"  平均 (Mean)      : {avg_latency:.3f}")
print(f"  标准差 (Std)     : {np.std(plot_latency):.3f}")
print(f"  最小值 (Min)     : {np.min(plot_latency):.3f}")
print(f"  最大值 (Max)     : {np.max(plot_latency):.3f}")
print(f"  百分位数:")
for p, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], percentiles):
    print(f"    P{p:2d}             : {v:.3f} ms")

if warmup_frames > 0:
    warmed_up = latency_ms[:warmup_frames]
    print(f"\n  CUDA 预热帧延迟: {warmed_up}")