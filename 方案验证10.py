import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.utils import scatter, add_self_loops, remove_self_loops
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. RD 外挂模块
# ============================================================
class RD_Plugin(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diff_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h_features, edge_index):
        row, col = edge_index
        h_diff = torch.abs(h_features[col] - h_features[row])
        diff_feat = self.diff_mlp(h_diff)
        
        node_diff = scatter(diff_feat, row, dim=0, dim_size=h_features.size(0), reduce='mean')
        node_diff = torch.nan_to_num(node_diff, nan=0.0)
        
        gate = self.gate_mlp(h_features)
        return h_features + gate * node_diff

# ============================================================
# 2. 高保真 H2GCN 传播逻辑 (避开稀疏矩阵，使用 scatter)
# ============================================================
def gcn_norm(edge_index, num_nodes):
    """标准的 GCN 归一化 (A_norm = D^{-1/2} A D^{-1/2})"""
    row, col = edge_index
    deg = scatter(torch.ones_like(row, dtype=torch.float), row, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return norm

def propagate(x, edge_index, norm):
    row, col = edge_index
    out = scatter(x[col] * norm.unsqueeze(-1), row, dim=0, dim_size=x.size(0), reduce='sum')
    return out

class H2GCN_Base(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, k=2, dropout=0.5):
        super().__init__()
        self.k = k
        self.dropout = dropout
        
        # H2GCN 核心：自身特征变换
        self.w_embed = nn.Linear(in_dim, hidden_dim, bias=False)
        
        # 拼接后的分类器 (2^(k+1) - 1) * hidden_dim
        self.classifier = nn.Linear(((2 ** (self.k + 1)) - 1) * hidden_dim, out_dim)
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # 1. 预处理：分离自环和无自环的边
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index_no_loop, _ = remove_self_loops(edge_index)
        
        norm_no_loop = gcn_norm(edge_index_no_loop, num_nodes)
        norm_loop = gcn_norm(edge_index_loop, num_nodes)
        
        # 2. 初始嵌入
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.w_embed(x)
        
        # 3. 提取多跳特征并分离 (H2GCN 的精髓)
        rs = [x]
        for _ in range(self.k):
            # 1-hop 提取 (不带自环)
            r = propagate(rs[-1], edge_index_no_loop, norm_no_loop)
            r = F.relu(r)
            rs.append(r)
            
            # 2-hop 提取 (带自环)
            r = propagate(r, edge_index_loop, norm_loop)
            rs.append(r)
            
        # 4. 拼接所有特征
        h_base = torch.cat(rs, dim=1)
        return h_base

class H2GCN_RD(nn.Module):
    def __init__(self, feat_dim, hidden_dim, class_dim, dropout=0.5, use_rd=True):
        super().__init__()
        self.h2gcn = H2GCN_Base(feat_dim, hidden_dim, class_dim, k=2, dropout=dropout)
        self.use_rd = use_rd
        
        feature_dim = ((2 ** 3) - 1) * hidden_dim  # k=2 时，特征维度是 7 * hidden_dim
        
        if use_rd:
            self.rd_plugin = RD_Plugin(feature_dim)

    def forward(self, x, edge_index):
        # 1. 基础特征提取
        h_base = self.h2gcn(x, edge_index)
        
        # 2. 外挂修正
        if self.use_rd:
            h_base = self.rd_plugin(h_base, edge_index)
            
        # 3. 分类
        out = self.h2gcn.classifier(h_base)
        return out

# ============================================================
# 3. 训练与评估
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
# 4. 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

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
