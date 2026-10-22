"""
SRQA-GAT 验证代码 V3 (修复自环 Bug)
包含: 4维结构描述子, 串联残差架构, 熵正则化, 多异配数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree, to_networkx
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. 结构描述子计算 (升级为 4 维，并修复自环)
# ============================================================
def compute_structural_descriptors(edge_index, num_nodes):
    """计算 [度数, 聚类系数, PageRank, K-Core]，并标准化"""
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg = deg / (deg.max() + 1e-8)

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)
    
    # 修复：移除自环，否则 nx.core_number 会报错
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # 聚类系数
    cc = nx.clustering(G)
    cc_tensor = torch.tensor([cc[i] for i in range(num_nodes)], dtype=torch.float)
    
    # PageRank
    pr = nx.pagerank(G)
    pr_tensor = torch.tensor([pr[i] for i in range(num_nodes)], dtype=torch.float)
    
    # K-Core
    kc = nx.core_number(G)
    kc_tensor = torch.tensor([kc[i] for i in range(num_nodes)], dtype=torch.float)
    kc_tensor = kc_tensor / (kc_tensor.max() + 1e-8)

    # 拼接 [N, 4]
    desc = torch.stack([deg, cc_tensor, pr_tensor, kc_tensor], dim=-1)
    return desc


# ============================================================
# 2. SRQA 分支
# ============================================================
class SRQABranch(nn.Module):
    def __init__(self, dim, num_roles=8, num_heads=4):
        super().__init__()
        self.num_roles = num_roles
        self.role_mlp = nn.Sequential(
            nn.Linear(4, 32), # 输入为 4 维
            nn.ReLU(),
            nn.Linear(32, num_roles)
        )
        self.role_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.role_norm = nn.LayerNorm(dim)

    def forward(self, h, struct_desc):
        # 1. 角色分配: [N, K]
        role_logits = self.role_mlp(struct_desc)
        role_probs = F.softmax(role_logits, dim=-1)

        # 2. 角色表示: [K, dim]
        weight_sum = role_probs.sum(dim=0).unsqueeze(-1) + 1e-8
        role_repr = torch.matmul(role_probs.T, h) / weight_sum

        # 3. 角色间 self-attention
        role_repr_q = role_repr.unsqueeze(0)
        role_updated, _ = self.role_attn(role_repr_q, role_repr_q, role_repr_q)
        role_updated = self.role_norm(role_updated.squeeze(0))

        # 4. 分配回节点
        h_from_roles = torch.matmul(role_probs, role_updated)
        
        return h_from_roles, role_probs


# ============================================================
# 3. 完整模型 (串联式架构)
# ============================================================
class SRQAGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_roles=8,
                 num_heads=4, dropout=0.6, use_srqa=True):
        super().__init__()
        self.use_srqa = use_srqa
        
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)

        if use_srqa:
            self.srqa = SRQABranch(hidden_dim, num_roles, num_heads)

        self.dropout = dropout
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, struct_desc):
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h_A = self.gat2(h, edge_index)
        h_A = F.elu(h_A)

        role_probs = None
        if self.use_srqa:
            h_B, role_probs = self.srqa(h_A, struct_desc)
            # 残差相加：保留局部特征 + 融入角色全局特征
            h = h_A + h_B
        else:
            h = h_A

        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_layer(h)
        return out, role_probs


# ============================================================
# 4. 训练与评估
# ============================================================
def train_and_eval(dataset_name, use_srqa, num_roles=8, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='/tmp/' + dataset_name, name=dataset_name.lower())
    elif dataset_name in ['Texas', 'Cornell', 'Wisconsin']:
        dataset = WebKB(root='/tmp/WebKB', name=dataset_name.lower())
    elif dataset_name == 'Actor':
        dataset = Actor(root='/tmp/Actor')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data = dataset[0].to(device)

    # 修复二维 mask
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    struct_desc = compute_structural_descriptors(data.edge_index.cpu(), data.num_nodes).to(device)

    in_dim = dataset.num_features
    out_dim = dataset.num_classes

    model = SRQAGAT(in_dim, 128, out_dim, num_roles=num_roles,
                    num_heads=4, dropout=0.6, use_srqa=use_srqa).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0
    best_test = 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out, role_probs = model(data.x, data.edge_index, struct_desc)
        
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        if use_srqa and role_probs is not None:
            avg_role_probs = role_probs.mean(dim=0)
            entropy = - (avg_role_probs * torch.log(avg_role_probs + 1e-8)).sum()
            loss = loss - 0.1 * entropy
            
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out, _ = model(data.x, data.edge_index, struct_desc)
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
    print(f"Using device: {device}")

    datasets = ['Cora', 'Citeseer', 'Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 80)
    print(f"{'Dataset':<12} {'Method':<15} {'Test Acc (mean±std)':<25}")
    print("=" * 80)

    for ds in datasets:
        for use_srqa in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_srqa=use_srqa, num_roles=8, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "GAT + SRQA" if use_srqa else "GAT (baseline)"
            print(f"{ds:<12} {method:<15} {mean:.2f} ± {std:.2f}")

        print("-" * 80)
