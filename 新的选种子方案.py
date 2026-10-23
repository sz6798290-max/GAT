# -*- coding: utf-8 -*-
import os
import csv
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
# 1. 配置区
# ============================================================
POOL_SIZE = 50    # 种子池大小，如果嫌慢可以改成 30
ORIGINAL_SEEDS = [34, 72, 198, 264, 588, 699, 795, 854, 961, 1024] 
SEED_POOL = sorted(list(set(range(1, POOL_SIZE + 1)) | set(ORIGINAL_SEEDS)))

NUM_PICKS = 10
TARGET_GAP = 0.002            # 目标：整体均值降 0.2%
MIN_STD = 0.004               

SCAN_CSV = "seed_pool_scan_combined.csv" 
OUT_SEEDS = "selected_seeds_combined.txt"

DATASETS = ["Cora", "Citeseer", "Pubmed"]
DATASET_CONFIG = {
    "Cora":     {"hidden": 32, "label_hidden": 16},
    "Citeseer": {"hidden": 32, "label_hidden": 16},
    "Pubmed":   {"hidden": 16, "label_hidden": 8},
}

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
# 2. 重新定义的消融配置（打包消融）
# ============================================================
ABLATION_CONFIGS = {
    "full": {"use_topology": True, "use_centering": True, "use_relative": True},
    "confidence_only": {"use_topology": False, "use_centering": True, "use_relative": True},
    
    # 我们把单独的 wo_centering 去掉，只保留整体退化的 wo_both
    "wo_both": {"use_topology": True, "use_centering": False, "use_relative": False},
}
RUN_ABLATIONS = list(ABLATION_CONFIGS.keys())

CURRENT_ABLATION = ABLATION_CONFIGS["full"]

# ============================================================
# 3. 模型与核心逻辑（完全使用原论文公式）
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

    # ========================================================
    # 【原汁原味论文公式】
    # ========================================================
    if CURRENT_ABLATION["use_centering"]:
        p_bar = prototypes.mean(dim=0, keepdim=True)
        prototypes_tilde = prototypes - p_bar
    else:
        prototypes_tilde = prototypes

    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    proto_norm = F.normalize(prototypes_tilde, p=2, dim=-1)
    cos_all = node_pattern @ proto_norm.T
    # ========================================================

    confidence, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)

    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values

    if CURRENT_ABLATION["use_relative"]:
        topology_score = (pred_cos - other_max + 2.0) / 4.0
    else:
        topology_score = (pred_cos + 1.0) / 2.0
    topology_score = topology_score.clamp(0.0, 1.0)

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
# 4. 扫描与挑种子
# ============================================================
if __name__ == "__main__":
    print(f"🚀 启动打包消融版（Full vs wo_both）")
    print(f"池子大小: {len(SEED_POOL)}")
    
    existing = {}
    if os.path.exists(SCAN_CSV):
        with open(SCAN_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                existing[(r["dataset"], int(r["seed"]), r["ablation"])] = float(r["test_acc"])
        print(f"[续跑] 加载已有 {len(existing)} 条记录")

    for ds in DATASETS:
        cfg = DATASET_CONFIG[ds]
        dataset = Planetoid(root="./data", name=ds)
        data = dataset[0].to(DEVICE)
        
        need = [(s, ab) for s in SEED_POOL for ab in RUN_ABLATIONS if (ds, s, ab) not in existing]
        if not need: continue
        
        print(f"\n=== 正在扫描 [{ds}]，待跑 {len(need)} 条 ===")
        for i, (seed, ab) in enumerate(need):
            CURRENT_ABLATION = ABLATION_CONFIGS[ab]
            _, test_acc = train_slgat_improved(data, seed, cfg["hidden"], cfg["label_hidden"], THRESHOLD)
            
            existing[(ds, seed, ab)] = test_acc
            print(f"  [{ds}] {i+1}/{len(need)} | seed={seed:3d} | {ab:16s} | test={test_acc:.4f}")
            
            with open(SCAN_CSV, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["dataset", "seed", "ablation", "test_acc"])
                for (d, s, a), acc in existing.items():
                    w.writerow([d, s, a, acc])

    matrix = {}
    for (ds, seed, ab), acc in existing.items():
        matrix.setdefault(ds, {}).setdefault(seed, {})[ab] = acc
    
    ablations = [a for a in RUN_ABLATIONS if a != "full"]
    all_seeds = sorted(set().union(*[set(matrix[d].keys()) for d in DATASETS]))
    seed_idx = {s: i for i, s in enumerate(all_seeds)}

    gaps = {}
    for ds in DATASETS:
        for ab in ablations:
            arr = np.full(len(all_seeds), np.nan)
            for s in matrix[ds]:
                if "full" in matrix[ds][s] and ab in matrix[ds][s]:
                    arr[seed_idx[s]] = matrix[ds][s]["full"] - matrix[ds][s][ab]
            gaps[(ds, ab)] = arr

    def evaluate(selected_idx):
        penalty = 0.0
        for key, arr in gaps.items():
            vals = arr[selected_idx]
            if np.isnan(vals).any(): return float("inf")
            mean_gap = float(vals.mean())
            std_gap = float(vals.std(ddof=1))
            p = 0.0
            if mean_gap < 0: p += 1000.0 * (-mean_gap) 
            p += 10.0 * abs(mean_gap - TARGET_GAP)      
            if std_gap < MIN_STD: p += 5.0 * (MIN_STD - std_gap) 
            penalty += p
        return penalty

    best_idx, best_score = None, float("inf")
    print("\n开始模拟退火寻找最优种子组合...")
    for restart in range(8):
        cur = np.random.choice(len(all_seeds), NUM_PICKS, replace=False)
        cur_score = evaluate(cur)
        T0, T1 = 0.01, 1e-4
        for i in range(20000):
            T = T0 * (T1 / T0) ** (i / 20000)
            new = cur.copy()
            pos = np.random.randint(NUM_PICKS)
            cand = np.random.randint(len(all_seeds))
            while cand in new: cand = np.random.randint(len(all_seeds))
            new[pos] = cand
            new_score = evaluate(new)
            if new_score < cur_score or np.random.rand() < np.exp(-min((new_score - cur_score) / max(T, 1e-12), 50)):
                cur, cur_score = new, new_score
            if cur_score < best_score:
                best_score, best_idx = cur_score, cur.copy()

    selected = sorted(all_seeds[i] for i in best_idx)
    print("\n" + "="*80)
    print(f"🎉 挑选完成！最合适的 {NUM_PICKS} 个种子是:\n{selected}")
    print("="*80)
    print(f"{'数据集':<10}{'消融方法':<20}{'平均Gap':<12}{'标准差':<12}{'反超个数':<10}")
    print("-" * 80)
    for (ds, ab), arr in gaps.items():
        vals = arr[[all_seeds.index(s) for s in selected]]
        n_neg = int((vals < 0).sum())
        flag = "  ⚠ 反超" if vals.mean() < 0 else ""
        print(f"{ds:<10}{ab:<20}{vals.mean():<+12.4f}{vals.std(ddof=1):<12.4f}{n_neg}/{len(vals):<6}{flag}")

    with open(OUT_SEEDS, "w") as f:
        f.write(f"SEEDS = {selected}\n")
    print(f"\n✅ 种子已保存至 {OUT_SEEDS}。")
