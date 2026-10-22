import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. 核心创新：基于高斯分布（Wasserstein距离）的注意力层
# ============================================================
class WassersteinGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 预测均值 mu
        self.lin_mu = nn.Linear(in_channels, out_channels, bias=False)
        # 预测方差 sigma^2 (使用 Softplus 保证大于 0)
        self.lin_var = nn.Linear(in_channels, out_channels, bias=False)
        
        # 注意力参数
        self.att = nn.Parameter(torch.Tensor(1, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)
        
        # 温度系数，控制分布的敏感度
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        # 1. 计算均值和方差
        mu = self.lin_mu(x)
        var = F.softplus(self.lin_var(x)) + 1e-5  # 保证方差为正
        
        return self.propagate(edge_index, mu=mu, var=var)

    def message(self, mu_i, var_i, mu_j, var_j, edge_index_i):
        # 2. 计算 2-Wasserstein 距离 (衡量两个高斯分布的差异)
        # W^2 = ||mu_i - mu_j||^2 + ||sqrt(var_i) - sqrt(var_j)||^2
        diff_mu = (mu_i - mu_j) ** 2
        diff_var = (torch.sqrt(var_i) - torch.sqrt(var_j)) ** 2
        w_dist = (diff_mu + diff_var).sum(dim=-1, keepdim=True) # [E, 1]
        
        # 3. 将距离转换为注意力分数 (距离越近分数越高)
        # 用负号表示距离越小，注意力越大，再乘温度系数调节
        score = -self.temp * w_dist
        
        # 4. softmax 归一化
        alpha = softmax(score.squeeze(-1), edge_index_i)
        
        # 5. 聚合：传递邻居的均值 mu_j
        return mu_j * alpha.unsqueeze(-1)


# ============================================================
# 2. 模型封装
# ============================================================
class W_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, use_wasserstein=True, dropout=0.5):
        super().__init__()
        self.use_wasserstein = use_wasserstein
        
        if use_wasserstein:
            self.conv = WassersteinGATConv(in_dim, 64)
        else:
            # 使用官方标准 GAT 作为 baseline
            from torch_geometric.nn import GATConv
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
def train_and_eval(dataset_name, use_wasserstein, seed=0, device='cpu'):
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

    model = W_GAT(dataset.num_features, dataset.num_classes, use_wasserstein=use_wasserstein).to(device)
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

    print("=" * 70)
    print(f"{'Dataset':<12} {'Method':<20} {'Test Acc (mean±std)':<20}")
    print("=" * 70)

    for ds in datasets:
        for use_w in [False, True]:
            accs = []
            for s in seeds:
                acc = train_and_eval(ds, use_wasserstein=use_w, seed=s, device=device)
                accs.append(acc * 100)

            mean = np.mean(accs)
            std = np.std(accs)
            method = "W-GAT (Ours)" if use_w else "GAT (baseline)"
            
            print(f"{ds:<12} {method:<20} {mean:.2f} ± {std:.2f}")
        print("-" * 70)
