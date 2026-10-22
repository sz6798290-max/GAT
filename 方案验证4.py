"""
SRQA-GAT 验证代码 V4 (Role-Biased Attention: 角色偏置注意力)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import MessagePassing
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
# 2. 自定义的 角色偏置注意力层 (Role-Biased GAT)
# ============================================================
class RoleBiasedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_roles=8, use_srqa=True):
        super().__init__(aggr='add')
        self.use_srqa = use_srqa
        
        # 标准的 GAT 特征变换和打分
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.Tensor(2 * out_channels, 1))
        nn.init.xavier_uniform_(self.a)
        
        if self.use_srqa:
            # 结构角色分配 (输入 4 维，输出 K 个角色的概率)
            self.role_mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, num_roles)
            )
            # 可学习的角色权重参数 beta
            self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index, struct_desc):
        x = self.W(x)
        
        # 计算角色概率 [N, K]
        if self.use_srqa:
            role_probs = F.softmax(self.role_mlp(struct_desc), dim=-1)
        else:
            role_probs = None

        return self.propagate(edge_index, x=x, role_probs=role_probs)

    def message(self, x_i, x_j, edge_index, role_probs):
        # 1. 标准 GAT 注意力分数
        # e_ij = LeakyReLU(a^T [Wx_i || Wx_j])
        x_cat = torch.cat([x_i, x_j], dim=-1)
        score = F.leaky_relu(torch.matmul(x_cat, self.a).squeeze(-1), 0.2)

        # 2. 加入角色偏置项
        if self.use_srqa and role_probs is not None:
            row, col = edge_index
            # 计算节点 i 和 j 的角色内积 (相似度)
            # r_i · r_j
            role_sim = (role_probs[row] * role_probs[col]).sum(dim=-1)
            # 将角色相似度加入打分
            score = score + self.beta * role_sim

        # 3. 计算注意力权重 (softmax 归一化)
        alpha = softmax(score, edge_index[0])
        return x_j * alpha.unsqueeze(-1)


# 引入 torch_geometric 里的 softmax
from torch_geometric.utils import softmax


# ============================================================
# 3. 完整模型
# ============================================================
class SRQAGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_roles=8, dropout=0.6, use_srqa=True):
        super().__init__()
        self.dropout = dropout
        # 单层 GAT，直接在注意力里用角色信息
        self.gat = RoleBiasedGATConv(in_dim, hidden_dim, num_roles, use_srqa)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, struct_desc):
        h = self.gat(x, edge_index, struct_desc)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_layer(h)
        return out


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

    model = SRQAGAT(in_dim, 64, out_dim, num_roles=num_roles, dropout=0.6, use_srqa=use_srqa).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val = 0
    best_test = 0

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, struct_desc)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, struct_desc)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()

            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc

    # 打印学到的 beta 值，看看模型多看重角色信息
    beta_val = model.gat.beta.item() if use_srqa else None
    return best_test, beta_val


# ============================================================
# 5. 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    datasets = ['Cora', 'Citeseer', 'Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 85)
    print(f"{'Dataset':<12} {'Method':<15} {'Test Acc (mean±std)':<25} {'Beta value':<10}")
    print("=" * 85)

    for ds in datasets:
        for use_srqa in [False, True]:
            accs = []
            betas = []
            for s in seeds:
                acc, beta = train_and_eval(ds, use_srqa=use_srqa, num_roles=8, seed=s, device=device)
                accs.append(acc * 100)
                if beta is not None:
                    betas.append(beta)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "GAT + SRQA" if use_srqa else "GAT (baseline)"
            beta_str = f"{np.mean(betas):.3f}" if betas else "-"
            print(f"{ds:<12} {method:<15} {mean:.2f} ± {std:.2f}{'':<10} {beta_str:<10}")

        print("-" * 85)
