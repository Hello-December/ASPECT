
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 12), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='white', edgecolor='#2c3e50', text_color='#2c3e50', fontsize=12, fontweight='normal'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, color=text_color, fontweight=fontweight)

def draw_arrow(ax, x1, y1, x2, y2, color='#34495e'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                           arrowstyle='->', color=color, 
                           linewidth=2, mutation_scale=20)
    ax.add_patch(arrow)

# 输入部分
draw_box(ax, 0.5, 10, 2, 1, 'Image Sequence', color='#e8f4f8', edgecolor='#2980b9', fontsize=13, fontweight='bold')

# YOLO26部分
draw_box(ax, 3, 9.5, 3, 2, 'YOLO26', color='#fdebd0', edgecolor='#e67e22', fontsize=14, fontweight='bold')
draw_box(ax, 3.2, 10.7, 1.2, 0.5, 'Backbone', color='#fff3cd', edgecolor='#d39e00', fontsize=10)
draw_box(ax, 4.6, 10.7, 1.2, 0.5, 'Neck', color='#fff3cd', edgecolor='#d39e00', fontsize=10)
draw_box(ax, 3.2, 10, 2.6, 0.5, 'Head', color='#fff3cd', edgecolor='#d39e00', fontsize=10)

# 从YOLO26输出的箭头
draw_arrow(ax, 5, 9.5, 5, 8.5)
draw_arrow(ax, 5, 9.5, 7, 9.5)

# YOLO26的两个输出
draw_box(ax, 6, 8, 2.5, 1, 'Initial Keypoints', color='#d5f4e6', edgecolor='#27ae60', fontsize=11)
draw_box(ax, 7, 9, 2.5, 1, 'Feature Map\n(Layer 15)', color='#d5f4e6', edgecolor='#27ae60', fontsize=11)

# Transformer Refiner部分 - 主框
draw_box(ax, 10, 6, 4.5, 5, 'Transformer Refiner', color='#ebf5fb', edgecolor='#3498db', fontsize=14, fontweight='bold')

# Feature Preprocessor
draw_box(ax, 10.2, 9.5, 4.1, 1, 'Feature Preprocessor', color='#d6eaf8', edgecolor='#2980b9', fontsize=11)
draw_box(ax, 10.4, 9.7, 1.8, 0.5, 'Conv Layers', fontsize=9)
draw_box(ax, 12.3, 9.7, 1.8, 0.5, 'Pool + Flatten', fontsize=9)

# Transformer Encoder
draw_box(ax, 10.2, 8, 4.1, 1.2, 'Transformer Encoder', color='#d6eaf8', edgecolor='#2980b9', fontsize=11)
draw_box(ax, 10.4, 8.2, 1.2, 0.35, 'Input Proj', fontsize=8)
draw_box(ax, 11.7, 8.2, 1.2, 0.35, 'Pos Encoding', fontsize=8)
draw_box(ax, 13, 8.2, 1.2, 0.35, 'Encoder Layers', fontsize=8)
draw_box(ax, 11.2, 8.6, 2.3, 0.35, 'Multi-Head Attention × 3', fontsize=8)

# Attention Pooling
draw_box(ax, 10.2, 6.5, 4.1, 1.2, 'Attention Pooling', color='#d6eaf8', edgecolor='#2980b9', fontsize=11)
draw_box(ax, 10.4, 6.7, 1.8, 0.5, 'Multi-Head Attention', fontsize=9)
draw_box(ax, 12.3, 6.7, 1.8, 0.5, 'Residual + LN', fontsize=9)

# Keypoint Head
draw_box(ax, 11.5, 5.2, 1.5, 1, 'Keypoint Head', color='#fadbd8', edgecolor='#c0392b', fontsize=11)
draw_box(ax, 11.7, 5.4, 1.1, 0.25, 'FC Layers', fontsize=8)
draw_box(ax, 11.7, 5.1, 1.1, 0.25, 'Sigmoid', fontsize=8)

# 箭头连接
draw_arrow(ax, 9.5, 9.5, 10, 9.5)
draw_arrow(ax, 12.25, 8, 12.25, 6.5)
draw_arrow(ax, 12.25, 6.5, 12.25, 5.2)
draw_arrow(ax, 12.25, 4.2, 12.25, 3.2)

#  refined keypoints
draw_box(ax, 10.5, 2.5, 3.5, 1, 'Refined Keypoints', color='#d5f4e6', edgecolor='#27ae60', fontsize=11)

# Hybrid Pose Estimation
draw_box(ax, 4, 1.5, 8, 1.5, 'Hybrid Pose Estimation\n(PnP Solver)', color='#e8daef', edgecolor='#8e44ad', fontsize=13, fontweight='bold')
draw_box(ax, 4.5, 1.7, 2, 0.4, 'Initial Keypoints', fontsize=9)
draw_box(ax, 7, 1.7, 2, 0.4, 'Refined Keypoints', fontsize=9)
draw_box(ax, 9.5, 1.7, 2, 0.4, 'PnP', fontsize=9)

# 最终输出
draw_box(ax, 6, 0.2, 4, 1, '6-DoF Pose\n(Rotation + Translation)', color='#e8f8f5', edgecolor='#16a085', fontsize=12, fontweight='bold')

# 连接到Hybrid Pose Estimation的箭头
draw_arrow(ax, 7.25, 8, 7.25, 3)
draw_arrow(ax, 12.25, 2.5, 12.25, 3)
draw_arrow(ax, 8, 1.5, 8, 0.2)

# 添加标题
ax.text(8, 11.5, 'ASPECT: Hybrid Pose Estimation System', ha='center', va='center', 
        fontsize=16, fontweight='bold', color='#2c3e50')

# 添加图例或说明
notes = [
    "Note: Transformer Refiner is the main contribution of this work.",
    "YOLO26 provides initial keypoints and features, while Transformer Refiner refines the keypoints.",
    "The hybrid approach combines both initial and refined keypoints for robust pose estimation."
]

for i, note in enumerate(notes):
    ax.text(0.5, 0.05 - i*0.03, note, ha='left', va='bottom', 
            fontsize=10, color='#7f8c8d', transform=ax.transAxes)

plt.tight_layout()
output_path = '/home/chenxuyang/PythonProjects/ASPECT/files/paper/system_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"System diagram saved to: {output_path}")
plt.show()

