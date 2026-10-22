import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GATv2Conv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 最终版：自适应残差差分 GAT (Adaptive RD-GAT)
# ============================================================
class Adaptive_RD_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_rd=True, dropout=0.5):
        super().__init__()
        self.use_rd = use_rd
        self.dropout = dropout
        
        # 特征变换
        self.lin = nn.Linear(in_dim, hidden_dim)
        
        # 官方 GATv2
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        if use_rd:
            # 1. 差分特征映射（把高维差值映射为标量）
            self.diff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            # 2. 尺度稳定（防止在 Actor 这种稠密图上梯度爆炸）
            self.diff_norm = nn.LayerNorm(1)
            # 3. 自适应门控（只看自身特征）
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.lin(x)
        
        # 标准 GATv2 输出
        h_out = self.gat(h, edge_index)
        
        if self.use_rd:
            row, col = edge_index
            # 1. 提取绝对差分
            h_diff = torch.abs(h[col] - h[row])
            diff_feat = self.diff_mlp(h_diff) # [E, 1]
            
            # 2. 均值聚合（第二版用的均值，效果最好）
            node_diff = torch.zeros(h.size(0), 1, device=x.device)
            node_diff.index_add_(0, row, diff_feat)
            node_count = torch.bincount(row, minlength=h.size(0)).unsqueeze(-1).float() + 1e-8
            node_diff = node_diff / node_count # [N, 1]
            
            # 3. 核心修复：方差归一化
            node_diff = self.diff_norm(node_diff)
            
            # 4. 门控（基于节点自身特征）
            gate = self.gate_mlp(h) # [N, 1]
            
            # 5. 融合
            h_out = h_out + gate * node_diff

        h_out = F.elu(h_out)
        h_out = F.dropout(h_out, p=self.dropout, training=self.training)
        out = self.out_layer(h_out)
        return out


# ============================================================
# 训练与评估
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

    model = Adaptive_RD_GAT(dataset.num_features, 64, dataset.num_classes, use_rd=use_rd).to(device)
    # 用 0.01 学习率，配合 LayerNorm 非常稳定
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
# 主程序
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 70)
    print(f"{'Dataset':<12} {'Method':<25} {'Test Acc (mean±std)':<20}")
    print("=" * 70)

    for ds in datasets:
        for use_rd in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_rd=use_rd, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "Adaptive RD-GAT (Ours)" if use_rd else "GATv2 (baseline)"
            
            print(f"{ds:<12} {method:<25} {mean:.2f} ± {std:.2f}")
        print("-" * 70)
