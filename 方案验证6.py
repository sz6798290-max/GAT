import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GATv2Conv, MessagePassing
from torch_geometric.utils import softmax
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 核心模型：注意力加权的残差差分 (Attention-Weighted RD)
# ============================================================
class Final_RD_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_rd=True, dropout=0.5):
        super().__init__()
        self.use_rd = use_rd
        self.dropout = dropout
        
        self.lin = nn.Linear(in_dim, hidden_dim)
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        if use_rd:
            # 差分特征的映射
            self.diff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid() # 输出一个 0~1 的差异权重
            )
            # 门控：决定在多大程度上把差分信息注入节点
            self.gate_mlp = nn.Linear(hidden_dim * 2, 1)
        
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.lin(x)
        
        # 1. 先用 GATv2 算出基础聚合结果
        h_out = self.gat(h, edge_index)
        
        if self.use_rd:
            row, col = edge_index
            
            # 2. 提取差分特征
            h_diff = torch.abs(h[col] - h[row])
            diff_weight = self.diff_mlp(h_diff) # [E, 1]
            
            # 3. 使用注意力或均值聚合差分（这里我们采用直接聚合）
            # 为了防止大数定律，这里用 max 聚合而不是 mean
            # PyG 的 scatter 聚合 (max) 可以极大缓解稠密图的噪声问题
            from torch_geometric.utils import scatter
            node_diff = scatter(diff_weight, row, dim=0, dim_size=h.size(0), reduce='max') # [N, 1]
            # 处理没有邻居的孤立节点
            node_diff = torch.nan_to_num(node_diff, nan=0.0)
            
            # 4. 用自适应门控融合
            # 把 h_out 和 差分上下文拼起来，让模型决定信多少
            cat_feat = torch.cat([h_out, node_diff.repeat(1, h_out.size(-1))], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(cat_feat))
            
            # 5. 加回网络（通过残差注入）
            h_out = h_out + gate * node_diff

        h_out = F.elu(h_out)
        h_out = F.dropout(h_out, p=self.dropout, training=self.training)
        out = self.out_layer(h_out)
        return out

# ============================================================
# 训练与评估 (完全不变)
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

    model = Final_RD_GAT(dataset.num_features, 64, dataset.num_classes, use_rd=use_rd).to(device)
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
            method = "Final RD-GAT (Ours)" if use_rd else "GATv2 (baseline)"
            
            print(f"{ds:<12} {method:<25} {mean:.2f} ± {std:.2f}")
        print("-" * 70)
