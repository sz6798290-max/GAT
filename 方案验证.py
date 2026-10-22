"""
SRQA-GAT 最小验证代码
数据集: Cora (同配), Chameleon (异配)
对比: GAT vs GAT + SRQA
目标: 看 SRQA 分支在异配图上是否有信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 结构描述子计算
# ============================================================
def compute_structural_descriptors(edge_index, num_nodes):
    """计算每个节点的 [度数, 聚类系数]，并标准化"""
    # 度数
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg = deg / (deg.max() + 1e-8)

    # 聚类系数 (用 networkx 算)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)
    cc = nx.clustering(G)
    cc_tensor = torch.tensor([cc[i] for i in range(num_nodes)], dtype=torch.float)

    # 拼接 [N, 2]
    desc = torch.stack([deg, cc_tensor], dim=-1)
    return desc


# ============================================================
# 2. SRQA 分支
# ============================================================
class SRQABranch(nn.Module):
    def __init__(self, in_dim, out_dim, num_roles=8, num_heads=4):
        super().__init__()
        self.num_roles = num_roles

        # 结构描述子 -> 角色分配
        self.role_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, num_roles)
        )

        # 节点特征变换
        self.node_proj = nn.Linear(in_dim, out_dim)

        # 角色间 self-attention (K 个角色之间)
        self.role_attn = nn.MultiheadAttention(
            out_dim, num_heads, batch_first=True
        )
        self.role_norm = nn.LayerNorm(out_dim)

    def forward(self, x, struct_desc):
        """
        x: [N, in_dim]
        struct_desc: [N, 2]
        """
        # 1. 角色分配: [N, K]
        role_logits = self.role_mlp(struct_desc)
        role_probs = F.softmax(role_logits, dim=-1)

        # 2. 节点特征变换: [N, out_dim]
        h = self.node_proj(x)

        # 3. 角色表示: [K, out_dim]
        weight_sum = role_probs.sum(dim=0).unsqueeze(-1) + 1e-8  # [K, 1]
        role_repr = torch.matmul(role_probs.T, h) / weight_sum      # [K, out_dim]

        # 4. 角色间 self-attention
        role_repr_q = role_repr.unsqueeze(0)  # [1, K, out_dim]
        role_updated, _ = self.role_attn(role_repr_q, role_repr_q, role_repr_q)
        role_updated = self.role_norm(role_updated.squeeze(0))  # [K, out_dim]

        # 5. 分配回节点: [N, out_dim]
        h_from_roles = torch.matmul(role_probs, role_updated)

        return h_from_roles


# ============================================================
# 3. 完整模型
# ============================================================
class SRQAGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_roles=8,
                 num_heads=4, dropout=0.5, use_srqa=True):
        super().__init__()
        self.use_srqa = use_srqa

        # GAT 分支
        self.gat = GATConv(in_dim, hidden_dim, heads=num_heads,
                           concat=False, dropout=dropout)

        if use_srqa:
            self.srqa = SRQABranch(in_dim, hidden_dim, num_roles, num_heads)
            self.gate = nn.Linear(hidden_dim * 2, 1)

        self.dropout = dropout
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, struct_desc):
        h_A = self.gat(x, edge_index)
        h_A = F.elu(h_A)

        if self.use_srqa:
            h_B = self.srqa(x, struct_desc)
            gate_input = torch.cat([h_A, h_B], dim=-1)
            g = torch.sigmoid(self.gate(gate_input))  # [N, 1]
            h = g * h_A + (1 - g) * h_B
        else:
            h = h_A
            g = None

        h = h + self.residual(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.out_layer(h)
        return h, g


# ============================================================
# 4. 训练与评估
# ============================================================
def train_and_eval(dataset_name, use_srqa, num_roles=8, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载数据
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='/tmp/' + dataset_name,
                                   name=dataset_name.lower())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = dataset[0].to(device)

    # 修复：Chameleon/Squirrel 的 mask 是 [N, 10]，取第一个划分
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    # 结构描述子
    struct_desc = compute_structural_descriptors(data.edge_index.cpu(), data.num_nodes).to(device)

    in_dim = dataset.num_features
    out_dim = dataset.num_classes

    model = SRQAGAT(in_dim, 64, out_dim, num_roles=num_roles,
                    num_heads=4, dropout=0.5, use_srqa=use_srqa).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    best_val = 0
    best_test = 0
    best_gate = None

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, struct_desc)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out, g = model(data.x, data.edge_index, struct_desc)
            pred = out.argmax(dim=1)

            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                if g is not None:
                    best_gate = g.mean().item()

    return best_test, best_gate


# ============================================================
# 5. 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    datasets = ['Cora', 'Chameleon']
    seeds = [0, 1, 2]

    print("=" * 80)
    print(f"{'Dataset':<12} {'Method':<15} {'Test Acc (mean±std)':<25} {'Gate mean':<10}")
    print("=" * 80)

    for ds in datasets:
        for use_srqa in [False, True]:
            accs = []
            gates = []
            for s in seeds:
                acc, gate = train_and_eval(ds, use_srqa=use_srqa,
                                           num_roles=8, seed=s, device=device)
                accs.append(acc * 100)
                if gate is not None:
                    gates.append(gate)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "GAT + SRQA" if use_srqa else "GAT (baseline)"
            gate_str = f"{np.mean(gates):.3f}" if gates else "-"
            print(f"{ds:<12} {method:<15} {mean:.2f} ± {std:.2f}{'':<10} {gate_str:<10}")

        print("-" * 80)
            gate_str = f"{np.mean(gates):.3f}" if gates else "-"
            print(f"{ds:<12} {method:<15} {mean:.2f} ± {std:.2f}{'':<10} {gate_str:<10}")

        print("-" * 80)
