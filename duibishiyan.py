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
# 配置
# ============================================================
DATASET = "Citeseer"
SEEDS = [0,1,2,3,4,5,6,7,8,9]
THRESHOLD = 0.7

HIDDEN = 32
LABEL_HIDDEN = 16

DROPOUT = 0.5
LR = 0.005
WEIGHT_DECAY = 5e-4
PRETRAIN_EPOCHS = 200
SELF_TRAIN_EPOCHS = 600
PATIENCE = 100
ALPHA = 0.5
BETA = 0.5
PSEUDO_TEMPERATURE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.getcwd(), "cora_old_vs_improved_thr07_10seeds")
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(logits, labels, mask):
    if int(mask.sum()) == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    return (pred == labels[mask]).float().mean().item()


def supervised_loss(logits, labels, mask):
    return F.cross_entropy(logits[mask], labels[mask])


def sharpen_probabilities(probs, temperature=PSEUDO_TEMPERATURE):
    probs = probs.clamp_min(1e-12)
    return F.softmax(torch.log(probs) / temperature, dim=-1)


# ============================================================
# SLGAT 模型（与之前一致）
# ============================================================
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
        e = F.leaky_relu((pair * self.att).sum(dim=-1), negative_slope=0.2)
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
        final_logits = 0.5 * (gcn_logits + att_logits)
        return final_logits, gcn_logits, att_logits


# ============================================================
# 邻居标签分布
# ============================================================
def _build_neighbor_dist(data, probs, num_classes):
    num_nodes = data.num_nodes
    src, dst = data.edge_index
    label_signal = probs.detach().clone()
    true_onehot = F.one_hot(data.y, num_classes=num_classes).to(probs.dtype)
    label_signal[data.train_mask] = true_onehot[data.train_mask]
    node_ids = torch.arange(num_nodes, device=probs.device)
    src_all = torch.cat([src, node_ids])
    dst_all = torch.cat([dst, node_ids])
    sums = scatter(label_signal[src_all], dst_all, dim=0,
                   dim_size=num_nodes, reduce="sum")
    counts = scatter(torch.ones(dst_all.numel(), device=probs.device),
                     dst_all, dim=0, dim_size=num_nodes,
                     reduce="sum").clamp_min(1.0)
    return sums / counts.unsqueeze(-1)


# ============================================================
# 旧方案：绝对余弦 + 等权原型
# ============================================================
def calculate_topology_scores_old(data, probs):
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes, device=probs.device)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(
                torch.tensor(c, device=probs.device), num_classes=num_classes
            ).float()
    prototypes = F.normalize(prototypes, p=2, dim=-1)

    confidence, predicted = probs.max(dim=-1)
    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    predicted_prototype = prototypes[predicted]
    topology_score = (node_pattern * predicted_prototype).sum(dim=-1).clamp(0.0, 1.0)
    fused_score = ALPHA * confidence + BETA * topology_score
    return predicted, confidence, topology_score, fused_score


# ============================================================
# 新方案：中心化原型 + 相对分数
# ============================================================
def calculate_topology_scores_improved(data, probs):
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes, device=probs.device)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(
                torch.tensor(c, device=probs.device), num_classes=num_classes
            ).float()

    p_bar = prototypes.mean(dim=0, keepdim=True)
    prototypes_tilde = prototypes - p_bar

    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    proto_norm = F.normalize(prototypes_tilde, p=2, dim=-1)
    cos_all = node_pattern @ proto_norm.T

    confidence, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)

    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values

    topology_score = (pred_cos - other_max + 1.0) / 2.0
    topology_score = topology_score.clamp(0.0, 1.0)
    fused_score = ALPHA * confidence + BETA * topology_score
    return predicted, confidence, topology_score, fused_score


# ============================================================
# 训练
# ============================================================
def train_slgat(data, seed, mode, threshold):
    set_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, HIDDEN, num_classes,
                  DROPOUT, label_hidden_dim=LABEL_HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 阶段1：监督预训练
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
        if patience_counter >= PATIENCE:
            break
    model.load_state_dict(best_state)

    # 阶段2：自训练
    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    selected_count = 0
    unlabeled_pool = ~data.train_mask

    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1)

            if mode == "old":
                predicted, _, _, fused_score = calculate_topology_scores_old(data, probs)
            elif mode == "improved":
                predicted, _, _, fused_score = calculate_topology_scores_improved(data, probs)
            else:
                raise ValueError("unknown mode: %s" % mode)

            selected_mask = unlabeled_pool & (fused_score >= threshold)
            selected_count = int(selected_mask.sum().item())
            soft_targets = sharpen_probabilities(probs).detach()

        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(data.x, data.edge_index)
        loss_sup = supervised_loss(logits, data.y, data.train_mask)
        if selected_mask.any():
            log_probs = F.log_softmax(logits[selected_mask], dim=-1)
            loss_unsup = -(soft_targets[selected_mask] * log_probs).sum(dim=-1).mean()
        else:
            loss_unsup = torch.tensor(0.0, device=DEVICE)
        loss = loss_sup + loss_unsup
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
        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
    return best_val, test_acc, selected_count


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 80)
    print("Cora | old vs improved | threshold = %.2f | 10 seeds" % THRESHOLD)
    print("device:", DEVICE)
    print("=" * 80)

    dataset = Planetoid(root="./data", name=DATASET)
    data = dataset[0].to(DEVICE)

    deg = torch.zeros(data.num_nodes, device=DEVICE)
    deg.scatter_add_(0, data.edge_index[1],
                     torch.ones_like(data.edge_index[1], dtype=torch.float))
    print("nodes=%d, train=%d, val=%d, test=%d, isolated(deg<=1)=%d (%.1f%%)" % (
        data.num_nodes, int(data.train_mask.sum()),
        int(data.val_mask.sum()), int(data.test_mask.sum()),
        int((deg <= 1).sum().item()),
        100.0 * int((deg <= 1).sum().item()) / data.num_nodes))

    methods = [
        ("SLGAT + old topology", "old"),
        ("SLGAT + improved",     "improved"),
    ]

    results = []
    t_all = time.time()

    for seed in SEEDS:
        print("\n--- seed=%d ---" % seed)
        for name, mode in methods:
            t0 = time.time()
            val_acc, test_acc, sel = train_slgat(data, seed, mode, THRESHOLD)
            dt = time.time() - t0
            print("  %-22s | Val=%.4f | Test=%.4f | Sel=%-6d | %.1fs" % (
                name, val_acc, test_acc, sel, dt))
            results.append({
                "seed": seed,
                "method": name,
                "mode": mode,
                "threshold": THRESHOLD,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "selected_pseudo_nodes": sel,
            })

    print("\nTotal time: %.1f min" % ((time.time() - t_all) / 60.0))

    # 保存逐次结果
    per_run_path = os.path.join(OUT_DIR, "results_per_run.csv")
    with open(per_run_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "method", "mode", "threshold",
            "val_acc", "test_acc", "selected_pseudo_nodes"
        ])
        writer.writeheader()
        writer.writerows(results)

    # 汇总
    print("\n" + "=" * 80)
    print("SUMMARY (Cora, threshold=%.2f, %d seeds)" % (THRESHOLD, len(SEEDS)))
    print("=" * 80)

    summary_rows = []
    for name, mode in methods:
        group = [r for r in results if r["method"] == name]
        val_values = np.array([r["val_acc"] for r in group])
        test_values = np.array([r["test_acc"] for r in group])
        print("\n%s" % name)
        print("  Val  acc: %.4f +/- %.4f" % (val_values.mean(), val_values.std(ddof=1)))
        print("  Test acc: %.4f +/- %.4f" % (test_values.mean(), test_values.std(ddof=1)))
        for r in group:
            print("    seed=%4d: val=%.4f test=%.4f sel=%d" % (
                r["seed"], r["val_acc"], r["test_acc"], r["selected_pseudo_nodes"]))
        summary_rows.append({
            "method": name,
            "mode": mode,
            "threshold": THRESHOLD,
            "val_mean": val_values.mean(),
            "val_std": val_values.std(ddof=1),
            "test_mean": test_values.mean(),
            "test_std": test_values.std(ddof=1),
        })

    summary_path = os.path.join(OUT_DIR, "results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "mode", "threshold",
            "val_mean", "val_std", "test_mean", "test_std"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nper-run saved to:", per_run_path)
    print("summary saved to:", summary_path)


if __name__ == "__main__":
    main()
