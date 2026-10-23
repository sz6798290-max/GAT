# -*- coding: utf-8 -*-
import os
import copy
import random
import time
import numpy as np
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax

try:
    from torch_geometric.utils import scatter
except ImportError:
    from torch_scatter import scatter

# ============================================================
# 1. 实验配置
# ============================================================
DATASETS = ["Cora", "Citeseer", "Pubmed"]

DATASET_CONFIG = {
    "Cora":     {"hidden": 32, "label_hidden": 16},
    "Citeseer": {"hidden": 32, "label_hidden": 16},
    "Pubmed":   {"hidden": 16, "label_hidden": 8},
}

# 固定使用你的 10 个种子
SEEDS = [34, 72, 198, 264, 588, 699, 795, 854, 961, 1024]

DROPOUT = 0.5
LR = 0.005
WEIGHT_DECAY = 5e-4

PRETRAIN_EPOCHS = 200
SELF_TRAIN_EPOCHS = 600
PATIENCE = 100

ALPHA = 0.5
BETA = 0.5
THRESHOLD = 0.8
PSEUDO_TEMPERATURE = 0.1
LAMBDA_U = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("环境初始化完成！当前使用设备:", DEVICE)
print("=" * 80)

# ============================================================
# 2. 基础模块与模型
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy(logits, labels, mask):
    if int(mask.sum().item()) == 0: return 0.0
    pred = logits[mask].argmax(dim=-1)
    return (pred == labels[mask]).float().mean().item()

def supervised_loss(logits, labels, mask):
    return F.cross_entropy(logits[mask], labels[mask])

def sharpen_probabilities(probs, temperature=PSEUDO_TEMPERATURE):
    probs = probs.clamp_min(1e-12)
    return F.softmax(torch.log(probs) / temperature, dim=-1)

class LabelGuidedAttention(nn.Module):
    def __init__(self, in_dim, label_dim, out_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.x_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.g_proj = nn.Linear(label_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(2 * out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))

    def forward(self, x, label_repr, edge_index):
        src, dst = edge_index
        h = self.x_proj(x)
        g = self.g_proj(label_repr)
        pair = torch.cat([g[dst], h[src]], dim=-1)
        e = torch.tanh((pair * self.att).sum(dim=-1))
        alpha = softmax(e, dst, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        messages = alpha.unsqueeze(-1) * h[src]
        out = scatter(messages, dst, dim=0, dim_size=x.size(0), reduce="sum")
        return out + self.bias

class SLGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout, label_hidden_dim=16):
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)
        self.label_conv1 = GCNConv(num_classes, label_hidden_dim)
        self.label_conv2 = GCNConv(label_hidden_dim, label_hidden_dim)
        self.att1 = LabelGuidedAttention(in_dim, label_hidden_dim, hidden_dim, dropout)
        self.att2 = LabelGuidedAttention(hidden_dim, label_hidden_dim, num_classes, dropout)

    def forward(self, x, edge_index):
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.relu(self.gcn1(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        gcn_logits = self.gcn2(z, edge_index)
        soft_labels = F.softmax(gcn_logits, dim=-1)
        label_repr = F.relu(self.label_conv1(soft_labels, edge_index))
        label_repr = F.dropout(label_repr, p=self.dropout, training=self.training)
        label_repr = self.label_conv2(label_repr, edge_index)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.att1(h, label_repr, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        att_logits = self.att2(h, label_repr, edge_index)
        return 0.5 * (gcn_logits + att_logits), gcn_logits, att_logits

def _build_neighbor_dist(data, probs, num_classes):
    num_nodes = data.num_nodes
    src, dst = data.edge_index
    label_signal = probs.detach().clone()
    true_onehot = F.one_hot(data.y, num_classes=num_classes).to(dtype=probs.dtype)
    label_signal[data.train_mask] = true_onehot[data.train_mask]
    node_ids = torch.arange(num_nodes, device=probs.device)
    src_all = torch.cat([src, node_ids])
    dst_all = torch.cat([dst, node_ids])
    sums = scatter(label_signal[src_all], dst_all, dim=0, dim_size=num_nodes, reduce="sum")
    counts = scatter(torch.ones(dst_all.numel(), device=probs.device, dtype=probs.dtype),
                     dst_all, dim=0, dim_size=num_nodes, reduce="sum").clamp_min(1.0)
    return sums / counts.unsqueeze(-1)

def calculate_topology_scores(data, probs, method="original"):
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes, device=probs.device, dtype=probs.dtype)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(torch.tensor(c, device=probs.device), num_classes=num_classes).to(dtype=probs.dtype)

    if method == "original":
        p_bar = prototypes.mean(dim=0, keepdim=True)
        prototypes_tilde = prototypes - p_bar
        node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
        proto_norm = F.normalize(prototypes_tilde, p=2, dim=-1)
        cos_all = node_pattern @ proto_norm.T
    elif method == "new":
        global_bias = neighbor_dist[data.train_mask].mean(dim=0, keepdim=True)
        node_tilde = neighbor_dist - global_bias
        prototypes_tilde = prototypes - global_bias
        cos_all = node_tilde @ prototypes_tilde.T

    confidence, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)

    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values

    if method == "original":
        topology_score = (pred_cos - other_max + 2.0) / 4.0
        topology_score = topology_score.clamp(0.0, 1.0)
    else:
        topology_score = torch.sigmoid(pred_cos - other_max)

    fused_score = ALPHA * confidence + BETA * topology_score
    return predicted, fused_score

def train_slgat(data, seed, hidden, label_hidden, threshold=0.8, method="original"):
    set_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, num_classes, DROPOUT, label_hidden_dim=label_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    best_state = None
    patience_counter = 0

    # 阶段一：预训练
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data.x, data.edge_index)
        loss = supervised_loss(logits, data.y, data.train_mask)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE: break

    if best_state is None:
        print("  [警告] 预训练未更新，使用初始权重。")
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    
    # 阶段二：自训练
    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    unlabeled_pool = ~data.train_mask

    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1)
            _, fused_score = calculate_topology_scores(data, probs, method=method)
            selected_mask = unlabeled_pool & (fused_score >= threshold)
            soft_targets = sharpen_probabilities(probs).detach()

        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data.x, data.edge_index)
        loss_sup = supervised_loss(logits, data.y, data.train_mask)

        if selected_mask.any():
            log_probs = F.log_softmax(logits[selected_mask], dim=-1)
            loss_unsup = -(soft_targets[selected_mask] * log_probs).sum(dim=-1).mean()
        else:
            loss_unsup = logits.sum() * 0.0

        loss = loss_sup + LAMBDA_U * loss_unsup
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)
            val_acc = accuracy(logits, data.y, data.val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
        
    return test_acc


# ============================================================
# 3. 主程序：执行 A/B 测试 (绝对强制的执行入口)
# ============================================================
def run_experiment():
    print(">>> 进入主程序，开始加载数据集...")
    for ds in DATASETS:
        print(f"\n[{ds}] 正在准备数据集...")
        cfg = DATASET_CONFIG[ds]
        dataset = Planetoid(root="./data", name=ds)
        data = dataset[0].to(DEVICE)
        print(f"[{ds}] 数据集加载完毕，节点数: {data.num_nodes}")

        orig_accs = []
        new_accs = []

        for seed in SEEDS:
            print(f"  --> 正在跑 seed={seed}...")
            t0 = time.time()
            
            orig_acc = train_slgat(data, seed, cfg["hidden"], cfg["label_hidden"], THRESHOLD, method="original")
            orig_accs.append(orig_acc)
            
            new_acc = train_slgat(data, seed, cfg["hidden"], cfg["label_hidden"], THRESHOLD, method="new")
            new_accs.append(new_acc)
            
            print(f"      seed={seed:4d} | 原方案={orig_acc:.4f} | 新方案={new_acc:.4f} | 提升={(new_acc-orig_acc)*100:+.2f}% | 耗时={time.time()-t0:.1f}s")

        orig_accs = np.array(orig_accs)
        new_accs = np.array(new_accs)

        print("\n" + "-" * 60)
        print(f"[{ds}] 最终对比结果 (10个种子):")
        print(f"原方案: {orig_accs.mean():.4f} ± {orig_accs.std(ddof=1):.4f}")
        print(f"新方案: {new_accs.mean():.4f} ± {new_accs.std(ddof=1):.4f}")
        print(f"平均提升: +{(new_accs.mean() - orig_accs.mean())*100:.2f}%")
        print("-" * 60)

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print("\n" + "!" * 80)
        print("发生错误，详细信息如下：")
        traceback.print_exc()
        print("!" * 80)
