import matplotlib.pyplot as plt

# 纯英文绘图，不需要设置中文字体，完美兼容 Colab
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 默认支持的英文字体

# 数据
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
cora_acc = [81.10, 81.60, 81.74, 81.84, 79.76]
citeseer_acc = [72.70, 73.62, 74.16, 75.36, 70.22]
pubmed_acc = [80.72, 80.70, 81.06, 81.68, 81.38]

# 创建画布 (宽度4，高度3，非常适合双栏论文的单栏排版)
plt.figure(figsize=(5, 3.5), dpi=300)

# 绘制三条折线
plt.plot(thresholds, cora_acc, marker='o', markersize=6, linewidth=2, label='Cora', color='#1f77b4')
plt.plot(thresholds, citeseer_acc, marker='s', markersize=6, linewidth=2, label='CiteSeer', color='#ff7f0e')
plt.plot(thresholds, pubmed_acc, marker='^', markersize=6, linewidth=2, label='PubMed', color='#2ca02c')

# 标注最优阈值点 (0.8)
plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.6)
plt.scatter(0.8, 81.84, color='#1f77b4', s=80, edgecolors='black', zorder=5)
plt.scatter(0.8, 75.36, color='#ff7f0e', s=80, edgecolors='black', zorder=5)
plt.scatter(0.8, 81.68, color='#2ca02c', s=80, edgecolors='black', zorder=5)

# 设置图表元素
plt.xlabel('Threshold \(\delta\)', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=11)
plt.xticks(thresholds)
plt.legend(fontsize=9, loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)

# 限制 y 轴范围，让折线的起伏更明显（突出 0.9 的暴跌）
plt.ylim(68, 86)

# 保存图片 (论文一般需要 PDF 或高清 PNG)
plt.tight_layout()
plt.savefig('threshold_sensitivity.pdf', format='pdf', bbox_inches='tight')
plt.savefig('threshold_sensitivity.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
