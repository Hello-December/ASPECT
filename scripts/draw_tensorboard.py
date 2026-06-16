import os
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "files", "figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.unicode_minus": False,
})

TAG_NAME_CN = {
    "Loss/train": "训练损失",
    "Loss/train_total": "训练总损失",
    "Loss/val": "验证损失",
    "Loss/val_total": "验证总损失",
    "Learning Rate": "学习率",
}


def get_cn_name(tag):
    if tag in TAG_NAME_CN:
        return TAG_NAME_CN[tag]
    return tag


def get_safe_filename(tag):
    cn = get_cn_name(tag)
    safe = cn.replace("/", "_").replace(" ", "_")
    return safe


def load_tensorboard_scalars(log_dir):
    ea = EventAccumulator(log_dir, size_guidance={})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        data[tag] = {"steps": steps, "values": values, "wall_times": wall_times}
    return data


def plot_single_tag(tag, data, output_dir, fmt):
    d = data[tag]
    cn_name = get_cn_name(tag)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["steps"], d["values"], color="#2c7bb6", linewidth=1.5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel(cn_name)
    ax.set_title(cn_name)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    safe_name = get_safe_filename(tag)
    for ext in fmt:
        out_path = os.path.join(output_dir, f"{safe_name}.{ext}")
        fig.savefig(out_path, format=ext)
        print(f"已保存: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="将 TensorBoard 日志中的标量数据绘制为 SVG/PDF 矢量图")
    parser.add_argument("--log_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "files", "refiner",
                                             "D4PED_dynamics_refiner_v1.1_train_3_direct", "tensorboard"),
                        help="TensorBoard 日志目录路径")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="图片输出目录")
    parser.add_argument("--format", type=str, nargs="+", default=["svg", "pdf"],
                        help="输出格式 (svg, pdf, png 等)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"正在加载 TensorBoard 日志: {args.log_dir}")
    data = load_tensorboard_scalars(args.log_dir)

    if not data:
        print("未在日志目录中找到标量数据。")
        return

    tags = list(data.keys())
    print(f"找到 {len(tags)} 个标量标签: {tags}")

    for tag in tags:
        plot_single_tag(tag, data, args.output_dir, args.format)

    print(f"\n所有图片已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()