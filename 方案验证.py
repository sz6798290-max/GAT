import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 核心创新：低秩子空间对齐注意力层
# ============================================================
class LowRankAlignedGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, rank=8):
        super().__init__(aggr='add', node_dim=0)
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.a = nn.Parameter(torch.Tensor(2 * out_channels, 1))
        nn.init.xavier_uniform_(self.a)
        
        # 低秩投影矩阵 U 和 V (借鉴 LoRA 思想)
        self.U = nn.Parameter(torch.Tensor(out_channels, rank))
        self.V = nn.Parameter(torch.Tensor(out_channels, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        
        # 控制对齐通道权重，初始给 0.5
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index):
        x = self.W(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index_i):
        # 1. 计算共享低秩子空间 P = U V^T
        P = torch.matmul(self.U, self.V.T) 
        
        # 投影特征
        x_i_proj = torch.matmul(x_i, P.T)
        x_j_proj = torch.matmul(x_j, P.T)

        # 2. 原始注意力分数
        x_cat_orig = torch.cat([x_i, x_j], dim=-1)
        score_orig = F.leaky_relu(torch.matmul(x_cat_orig, self.a).squeeze(-1), 0.2)
        
        # 3. 对齐后的相似度分数
        score_aligned = (x_i_proj * x_j_proj).sum(dim=-1)
        
        # 4. 融合分数
        score = score_orig + self.gamma * score_aligned
        
        # 5. 归一化
        alpha = softmax(score, edge_index_i)
        return x_j * alpha.unsqueeze(-1)


# ============================================================
# 2. 模型封装
# ============================================================
class MyModel(nn.Module):
    def __init__(self, in_dim, out_dim, use_lowrank=True, dropout=0.6):
        super().__init__()
        self.use_lowrank = use_lowrank
        if use_lowrank:
            self.conv = LowRankAlignedGATConv(in_dim, 64, rank=8)
        else:
            self.conv = GATConv(in_dim, 64, heads=1, concat=False, dropout=dropout)
        
        self.dropout = dropout
        self.out_layer = nn.Linear(64, out_dim)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_layer(h)
        return out


# ============================================================
# 3. 训练与评估
# ============================================================
def train_and_eval(dataset_name, use_lowrank, seed=0, device='cpu'):
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

    model = MyModel(dataset.num_features, dataset.num_classes, use_lowrank=use_lowrank).to(device)
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

    gamma_val = model.conv.gamma.item() if use_lowrank else None
    return best_test, gamma_val


# ============================================================
# 4. 主程序：只跑异配图
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    seeds = [0, 1, 2]

    print("=" * 75)
    print(f"{'Dataset':<12} {'Method':<18} {'Test Acc (mean±std)':<22} {'Gamma':<10}")
    print("=" * 75)

    for ds in datasets:
        for use_lowrank in [False, True]:
            accs = []
            gammas = []
            for s in seeds:
                acc, gamma = train_and_eval(ds, use_lowrank=use_lowrank, seed=s, device=device)
                accs.append(acc * 100)
                if gamma is not None:
                    gammas.append(gamma)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "LowRank-Aligned" if use_lowrank else "GAT (baseline)"
            gamma_str = f"{np.mean(gammas):.3f}" if gammas else "-"
            
            print(f"{ds:<12} {method:<18} {mean:.2f} ± {std:.2f}{'':<6} {gamma_str:<10}")
        print("-" * 75)
