import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. RD 外挂模块（差分特征提取 + 自适应门控）
# ============================================================
class RD_Plugin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 差分特征映射
        self.diff_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        # 自适应门控
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_base, x, edge_index):
        row, col = edge_index
        # 计算绝对差分
        h_diff = torch.abs(x[col] - x[row])
        diff_feat = self.diff_mlp(h_diff)
        
        # 均值聚合差分
        node_diff = scatter(diff_feat, row, dim=0, dim_size=x.size(0), reduce='mean')
        node_diff = torch.nan_to_num(node_diff, nan=0.0)
        
        # 门控
        gate = self.gate_mlp(x)
        
        return h_base + gate * node_diff

# ============================================================
# 2. H2GCN 核心思想（基于 PyG 原生 GCNConv 的稳定实现）
# ============================================================
class StableH2GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        # H2GCN 需要分别提取 1-hop 和 2-hop 邻居
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 将自身特征 (x0)、1-hop (h1)、2-hop (h2) 拼接
        # 需要先对 x0 做维度映射，保证和 h1, h2 维度一致
        self.self_lin = nn.Linear(in_dim, hidden_dim)
        
        self.out_layer = nn.Linear(hidden_dim * 3, out_dim)

    def forward(self, x, edge_index):
        # 0-hop (自身特征)
        x0 = self.self_lin(x)
        
        # 1-hop
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        
        # 2-hop
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        
        # H2GCN 标志性的分离与拼接
        h_base = torch.cat([x0, h1, h2], dim=-1)
        
        # 基础分类输出
        out_base = self.out_layer(h_base)
        
        # 返回分类结果，以及隐藏特征（用于给 RD 模块提供差分计算的源）
        return out_base, h_base

# ============================================================
# 3. 完整模型（H2GCN + RD 外挂）
# ============================================================
class H2GCN_RD(nn.Module):
    def __init__(self, feat_dim, hidden_dim, class_dim, dropout=0.5, use_rd=True):
        super().__init__()
        self.h2gcn = StableH2GCN(feat_dim, hidden_dim, class_dim, dropout)
        self.use_rd = use_rd
        if use_rd:
            # 注意：H2GCN 拼接后的维度是 hidden_dim * 3
            self.rd_plugin = RD_Plugin(hidden_dim * 3)

    def forward(self, x, edge_index):
        out_base, h_base = self.h2gcn(x, edge_index)
        if self.use_rd:
            # 用 H2GCN 提取的强特征 h_base 来计算差分并注入
            out_base = self.rd_plugin(out_base, h_base, edge_index)
        return out_base

# ============================================================
# 4. 训练与评估
# ============================================================
def train_and_eval(dataset_name, use_rd, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='/tmp/' + dataset_name, name=dataset_name.lower())
    elif dataset_name == 'Actor':
        dataset = Actor(root='/tmp/Actor')
    elif dataset_name in ['Texas', 'Cornell']:
        dataset = WebKB(root='/tmp/WebKB', name=dataset_name.lower())

    data = dataset[0].to(device)

    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    model = H2GCN_RD(
        feat_dim=dataset.num_features,
        hidden_dim=64,
        class_dim=dataset.num_classes,
        dropout=0.5,
        use_rd=use_rd
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0
    best_test = 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

    return best_test

# ============================================================
# 5. 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2] # 先用3个种子快速跑通看趋势，有效果再加到5个

    print("=" * 75)
    print(f"{'Dataset':<12} {'Method':<22} {'Test Acc (mean±std)':<20}")
    print("=" * 75)

    for ds in datasets:
        for use_rd in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_rd=use_rd, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "H2GCN + RD (Ours)" if use_rd else "H2GCN (baseline)"

            print(f"{ds:<12} {method:<22} {mean:.2f} ± {std:.2f}")
        print("-" * 75)
