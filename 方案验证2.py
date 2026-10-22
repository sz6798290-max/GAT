import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GATConv
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 核心模型：引入 Null Node (空节点) 的 GAT
# ============================================================
class NullGuidedGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes, use_null=True, dropout=0.5):
        super().__init__()
        self.use_null = use_null
        self.num_nodes = num_nodes
        
        # 核心：定义全局唯一的“空节点”特征，作为可学习参数
        if use_null:
            self.null_token = nn.Parameter(torch.randn(1, in_dim))
            
        # 就用单层 GAT，不搞花里胡哨的
        self.gat = GATConv(in_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.out_layer = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        if self.use_null:
            N = x.size(0)
            # 1. 把空节点拼接到原特征矩阵最后一行 -> [N+1, in_dim]
            x_aug = torch.cat([x, self.null_token], dim=0)
            
            # 2. 构建虚拟边：每个节点 <-> 空节点
            null_edge = torch.tensor([[i for i in range(N)], [N] * N], device=x.device)
            null_edge_rev = torch.tensor([[N] * N, [i for i in range(N)]], device=x.device)
            
            # 拼接到原边上
            edge_index_aug = torch.cat([edge_index, null_edge, null_edge_rev], dim=1)
            
            # 3. 用扩充后的图进行 GAT 传播
            h_aug = self.gat(x_aug, edge_index_aug)
            
            # 4. 丢弃空节点的表示，只取原节点的
            h = h_aug[:N, :]
        else:
            h = self.gat(x, edge_index)

        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_layer(h)
        return out


# ============================================================
# 训练与评估
# ============================================================
def train_and_eval(dataset_name, use_null, seed=0, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='/tmp/' + dataset_name, name=dataset_name.lower())
    elif dataset_name == 'Actor':
        dataset = Actor(root='/tmp/Actor')
    elif dataset_name in ['Texas', 'Cornell']:
        dataset = WebKB(root='/tmp/WebKB', name=dataset_name.lower())
    
    data = dataset[0].to(device)

    # 处理 2D mask
    if data.train_mask.dim() == 2:
        data.train_mask = data.train_mask[:, 0]
        data.val_mask = data.val_mask[:, 0]
        data.test_mask = data.test_mask[:, 0]

    model = NullGuidedGAT(dataset.num_features, 64, dataset.num_classes, 
                          data.num_nodes, use_null=use_null, dropout=0.5).to(device)
    
    # 稍微调低学习率，让它更平稳
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
# 主程序：跑异配图
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 75)
    print(f"{'Dataset':<12} {'Method':<22} {'Test Acc (mean±std)':<22}")
    print("=" * 75)

    for ds in datasets:
        for use_null in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_null=use_null, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "Null-Guided GAT" if use_null else "GAT (baseline)"
            
            print(f"{ds:<12} {method:<22} {mean:.2f} ± {std:.2f}")
        print("-" * 75)
