import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果是英文论文，可以删掉这两行）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
cora_acc = [81.10, 81.60, 81.74, 81.84, 79.76]
citeseer_acc = [72.70, 73.62, 74.16, 75.36, 70.22]
pubmed_acc = [80.72, 80.70, 81.06, 81.68, 81.38]

# 创建画布
plt.figure(figsize=(8, 5), dpi=300)

# 绘制三条折线
plt.plot(thresholds, cora_acc, marker='o', markersize=8, linewidth=2.5, label='Cora', color='#1f77b4')
plt.plot(thresholds, citeseer_acc, marker='s', markersize=8, linewidth=2.5, label='CiteSeer', color='#ff7f0e')
plt.plot(thresholds, pubmed_acc, marker='^', markersize=8, linewidth=2.5, label='PubMed', color='#2ca02c')

# 标注最优阈值点 (0.8)
plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.6)
plt.scatter(0.8, 81.84, color='#1f77b4', s=120, edgecolors='black', zorder=5)
plt.scatter(0.8, 75.36, color='#ff7f0e', s=120, edgecolors='black', zorder=5)
plt.scatter(0.8, 81.68, color='#2ca02c', s=120, edgecolors='black', zorder=5)

# 设置图表元素
plt.title('Sensitivity Analysis of Threshold δ', fontsize=14, fontweight='bold')
plt.xlabel('Threshold δ', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(thresholds)
plt.legend(fontsize=11, loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)

# 调整 y 轴范围，让差异更明显（可选）
# plt.ylim(68, 86)

# 保存图片
plt.tight_layout()
plt.savefig('threshold_sensitivity.pdf', format='pdf')
plt.savefig('threshold_sensitivity.png', format='png', dpi=300)
plt.show()
