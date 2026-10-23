# -*- coding: utf-8 -*-
import os
import copy
import random
import time
import numpy as np

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
# 1. 实验配置 (固定使用你最初的 10 个种子)
# ============================================================
DATASETS = ["Cora", "Citeseer", "Pubmed"]
DATASET_CONFIG = {
    "Cora":     {"hidden": 32, "label_hidden": 16},
    "Citeseer": {"hidden": 32, "label_hidden": 16},
    "Pubmed":   {"hidden": 16, "label_hidden": 8},
}

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

# ============================================================
# 2. 精炼版消融配置 (只保留 3 组)
# ============================================================
ABLATION_CONFIGS = {
    "Full": {"use_topology": True, "use_centering": True, "use_relative": True},
    "w/o Topology": {"use_topology": False, "use_centering": True, "use_relative": True},
    "w/o Debiased-Matching": {"use_topology": True, "use_centering": False, "use_relative": False},
}
RUN_ABLATIONS = list(ABLATION_CONFIGS.keys())

CURRENT_ABLATION = ABLATION_CONFIGS["Full"]

# ============================================================
# 3. 模型与核心逻辑 (完全保留你的原论文公式)
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

def calculate_topology_scores_improved(data, probs):
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)
    prototypes = torch.zeros(num_classes, num_classes, device=probs.device, dtype=probs.dtype)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(torch.tensor(c, device=probs.device), num_classes=num_classes).to(dtype=probs.dtype)

    # 中心化开关 (去偏匹配模块的一部分)
    if CURRENT_ABLATION["use_centering"]:
        p_bar = prototypes.mean(dim=0, keepdim=True)
        prototypes_tilde = prototypes - p_bar
    else:
        prototypes_tilde = prototypes

    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    proto_norm = F.normalize(prototypes_tilde, p=2, dim=-1)
    cos_all = node_pattern @ proto_norm.T

    confidence, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)

    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values

    # 相对相似度开关 (去偏匹配模块的另一部分)
    if CURRENT_ABLATION["use_relative"]:
        topology_score = (pred_cos - other_max + 2.0) / 4.0
    else:
        topology_score = (pred_cos + 1.0) / 2.0
    topology_score = topology_score.clamp(0.0, 1.0)

    # 拓扑开关
    if CURRENT_ABLATION["use_topology"]:
        fused_score = ALPHA * confidence + BETA * topology_score
    else:
        fused_score = confidence

    return predicted, confidence, topology_score, fused_score

def train_slgat_improved(data, seed, hidden, label_hidden, threshold=0.8):
    set_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, num_classes, DROPOUT, label_hidden_dim=label_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val = -1.0
    best_state = None
    patience_counter = 0

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

    if best_state is None: raise RuntimeError("预训练失败")
    model.load_state_dict(best_state)

    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    unlabeled_pool = ~data.train_mask

    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1)
            _, _, _, fused_score = calculate_topology_scores_improved(data, probs)
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
    return best_val, test_acc

# ============================================================
# 4. 主程序：跑 3 组打包消融并输出表格
# ============================================================
if __name__ == "__main__":
    print(f"🚀 启动打包消融实验（3 组配置，原版 10 个种子）")
    
    results = {ds: {ab: [] for ab in RUN_ABLATIONS} for ds in DATASETS}
    total_runs = len(DATASETS) * len(SEEDS) * len(RUN_ABLATIONS)
    current_run = 0

    for ds in DATASETS:
        cfg = DATASET_CONFIG[ds]
        dataset = Planetoid(root="./data", name=ds)
        data = dataset[0].to(DEVICE)
        
        print(f"\n===== 正在测试数据集 [{ds}] =====")
        for seed in SEEDS:
            for ab in RUN_ABLATIONS:
                CURRENT_ABLATION = ABLATION_CONFIGS[ab]
                val_acc, test_acc = train_slgat_improved(data, seed, cfg["hidden"], cfg["label_hidden"], THRESHOLD)
                
                results[ds][ab].append(test_acc)
                current_run += 1
                print(f"  [{current_run}/{total_runs}] seed={seed:4d} | {ab:22s} | Test Acc = {test_acc:.4f}")

    # 打印最终 Markdown 表格
    print("\n\n" + "="*80)
    print("✅ 最终真实论文消融表格 (3 组打包消融):")
    print("="*80)
    
    for ds in DATASETS:
        print(f"\n### Dataset: {ds}")
        print(f"| Method | Test Accuracy (Mean ± Std) |")
        print(f"| :--- | :--- |")
        for ab in RUN_ABLATIONS:
            accs = np.array(results[ds][ab])
            mean_acc = accs.mean()
            std_acc = accs.std(ddof=1)
            bold = "**" if ab == "Full" else ""
            print(f"| {bold}{ab}{bold} | {bold}{mean_acc:.4f} ± {std_acc:.4f}{bold} |")
            
    print("\n实验结束。")
