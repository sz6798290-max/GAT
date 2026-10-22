import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB, Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree, scatter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 核心外挂：频域残差融合模块 (Spectral Residual Fusion)
# 理论：H_high = X - D^{-1/2} A D^{-1/2} X (近似高通滤波)
# ============================================================
class SpectralResidualFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 用于融合高频分量的自适应缩放门控
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        N = x.size(0)
        row, col = edge_index
        
        # 1. 计算图的度
        deg = degree(row, N, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        
        # 2. 计算低通平滑结果 (类似于 GCN 的传播)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        h_low = scatter(x[col] * norm.unsqueeze(-1), row, dim=0, dim_size=N, reduce='sum')
        
        # 3. 核心：提取高频残差 (原始特征 - 平滑特征)
        h_high = x - h_low
        
        # 4. 自适应门控：让模型自己决定在多大程度上利用高频信息
        g = self.gate(x)
        
        # 返回增强后的特征
        return x + g * h_high

# ============================================================
# 主干网络：GATv2 + 频域残差融合
# ============================================================
class SRF_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_srf=True, dropout=0.5):
        super().__init__()
        self.use_srf = use_srf
        self.lin = nn.Linear(in_dim, hidden_dim)
        
        # 外挂加在特征映射后
        if self.use_srf:
            self.srf = SpectralResidualFusion(hidden_dim)
            
        # 标准的 GAT 层
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # 特征映射
        h = self.lin(x)
        
        # 注入频域残差外挂
        if self.use_srf:
            h = self.srf(h, edge_index)
            
        # GAT 聚合
        h = F.elu(self.gat1(h, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        out = self.gat2(h, edge_index)
        return out

# ============================================================
# 训练与评估（严格跑 5 个种子）
# ============================================================
def train_and_eval(dataset_name, use_srf, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
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

    model = SRF_GAT(
        in_dim=dataset.num_features,
        hidden_dim=64,
        out_dim=dataset.num_classes,
        use_srf=use_srf
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

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
# 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Cora', 'Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2, 3, 4] # 满足 5 个种子的学术要求

    print("=" * 75)
    print(f"{'Dataset':<12} {'Method':<22} {'Test Acc (mean±std)':<20}")
    print("=" * 75)

    for ds in datasets:
        for use_srf in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_srf=use_srf, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "SRF-GAT (Ours)" if use_srf else "GAT (baseline)"

            print(f"{ds:<12} {method:<22} {mean:.2f} ± {std:.2f}")
        print("-" * 75)
