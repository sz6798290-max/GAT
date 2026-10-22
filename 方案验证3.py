"""
SRQA-GAT 验证代码 V3 (单层GAT + 并联门控 + 4维结构特征)
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
# 1. 结构描述子计算 (4 维)
# ============================================================
def compute_structural_descriptors(edge_index, num_nodes):
    deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    deg = deg / (deg.max() + 1e-8)

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    cc = nx.clustering(G)
    cc_tensor = torch.tensor([cc[i] for i in range(num_nodes)], dtype=torch.float)
    
    pr = nx.pagerank(G)
    pr_tensor = torch.tensor([pr[i] for i in range(num_nodes)], dtype=torch.float)
    
    kc = nx.core_number(G)
    kc_tensor = torch.tensor([kc[i] for i in range(num_nodes)], dtype=torch.float)
    kc_tensor = kc_tensor / (kc_tensor.max() + 1e-8)

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
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_roles)
        )
        self.role_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.role_norm = nn.LayerNorm(dim)

    def forward(self, h, struct_desc):
        role_logits = self.role_mlp(struct_desc)
        role_probs = F.softmax(role_logits, dim=-1)

        weight_sum = role_probs.sum(dim=0).unsqueeze(-1) + 1e-8
        role_repr = torch.matmul(role_probs.T, h) / weight_sum

        role_repr_q = role_repr.unsqueeze(0)
        role_updated, _ = self.role_attn(role_repr_q, role_repr_q, role_repr_q)
        role_updated = self.role_norm(role_updated.squeeze(0))

        h_from_roles = torch.matmul(role_probs, role_updated)
        return h_from_roles, role_probs


# ============================================================
# 3. 完整模型 (单层 GAT + 门控)
# ============================================================
class SRQAGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_roles=8,
                 num_heads=4, dropout=0.6, use_srqa=True):
        super().__init__()
        self.use_srqa = use_srqa
        
        # 退回单层 GAT
        self.gat = GATConv(in_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)

        if use_srqa:
            self.srqa = SRQABranch(hidden_dim, num_roles, num_heads)
            self.gate = nn.Linear(hidden_dim * 2, 1)

        self.dropout = dropout
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, struct_desc):
        h_A = self.gat(x, edge_index)
        h_A = F.elu(h_A)

        if self.use_srqa:
            h_B, role_probs = self.srqa(h_A, struct_desc)
            
            # 并联门控
            gate_input = torch.cat([h_A, h_B], dim=-1)
            g = torch.sigmoid(self.gate(gate_input))
            h = g * h_A + (1 - g) * h_B
        else:
            h = h_A
            g = None
            role_probs = None

        h = h + self.residual(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_layer(h)
        return out, g, role_probs


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
    best_gate = None

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out, _, _ = model(data.x, data.edge_index, struct_desc)
        
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out, g, _ = model(data.x, data.edge_index, struct_desc)
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

    datasets = ['Cora', 'Citeseer', 'Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 85)
    print(f"{'Dataset':<12} {'Method':<15} {'Test Acc (mean±std)':<25} {'Gate mean':<10}")
    print("=" * 85)

    for ds in datasets:
        for use_srqa in [False, True]:
            accs = []
            gates = []
            for s in seeds:
                acc, gate = train_and_eval(ds, use_srqa=use_srqa, num_roles=8, seed=s, device=device)
                accs.append(acc * 100)
                if gate is not None:
                    gates.append(gate)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "GAT + SRQA" if use_srqa else "GAT (baseline)"
            gate_str = f"{np.mean(gates):.3f}" if gates else "-"
            print(f"{ds:<12} {method:<15} {mean:.2f} ± {std:.2f}{'':<10} {gate_str:<10}")

        print("-" * 85)
