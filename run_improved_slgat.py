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
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import softmax
try:
    from torch_geometric.utils import scatter
except ImportError:
    from torch_scatter import scatter

# ============================================================
# 配置
# ============================================================
DATASETS = ["Cora", "Citeseer", "Pubmed"]

DATASET_CONFIG = {
    "Cora":     {"hidden": 32, "label_hidden": 16, "heads": 8},
    "Citeseer": {"hidden": 32, "label_hidden": 16, "heads": 8},
    "Pubmed":   {"hidden": 16, "label_hidden": 8,  "heads": 4},
}

SEEDS = [34,72,198,264,588,699,795,854,961,1024]

DROPOUT = 0.5
LR = 0.005
WEIGHT_DECAY = 5e-4
PRETRAIN_EPOCHS = 200
SELF_TRAIN_EPOCHS = 600
PATIENCE = 100
ALPHA = 0.5
BETA = 0.5
THRESHOLD = 0.8          # old 和 improved 统一用 0.7，保证公平对比
PSEUDO_TEMPERATURE = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_OUT_DIR = os.path.join(os.getcwd(), "slgat_full_compare_10seeds")
os.makedirs(ROOT_OUT_DIR, exist_ok=True)


# ============================================================
# 工具函数
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
# Vanilla GAT 基线
# ============================================================
class VanillaGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout, heads=8):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads,
                            concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, num_classes, heads=1,
                            concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)


def train_vanilla_gat(data, seed, hidden, heads):
    set_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = VanillaGAT(data.num_features, hidden, num_classes,
                       DROPOUT, heads=heads).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    best_val = -1.0
    best_state = None
    patience_counter = 0
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = supervised_loss(logits, data.y, data.train_mask)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
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
        logits = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
    return best_val, test_acc


# ============================================================
# 标签引导注意力层
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


# ============================================================
# SLGAT 模型
# ============================================================
class SLGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout,
                 label_hidden_dim=16):
        super().__init__()
        self.dropout = dropout
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)
        self.label_conv1 = GCNConv(num_classes, label_hidden_dim)
        self.label_conv2 = GCNConv(label_hidden_dim, label_hidden_dim)
        self.att1 = LabelGuidedAttention(in_dim, label_hidden_dim,
                                         hidden_dim, dropout)
        self.att2 = LabelGuidedAttention(hidden_dim, label_hidden_dim,
                                         num_classes, dropout)

    def forward(self, x, edge_index):
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.relu(self.gcn1(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        gcn_logits = self.gcn2(z, edge_index)
        soft_labels = F.softmax(gcn_logits, dim=-1)
        label_repr = F.relu(self.label_conv1(soft_labels, edge_index))
        label_repr = F.dropout(label_repr, p=self.dropout,
                               training=self.training)
        label_repr = self.label_conv2(label_repr, edge_index)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.att1(h, label_repr, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        att_logits = self.att2(h, label_repr, edge_index)
        final_logits = 0.5 * (gcn_logits + att_logits)
        return final_logits, gcn_logits, att_logits


# ============================================================
# 邻居标签分布（两个拓扑方案共用）
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
    neighbor_dist = sums / counts.unsqueeze(-1)
    return neighbor_dist


# ============================================================
# 原方案拓扑打分
# ============================================================
def calculate_topology_scores_old(data, probs):
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes, device=probs.device)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(
                torch.tensor(c, device=probs.device),
                num_classes=num_classes
            ).float()
    prototypes = F.normalize(prototypes, p=2, dim=-1)

    confidence, predicted = probs.max(dim=-1)
    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    predicted_prototype = prototypes[predicted]
    topology_score = (node_pattern * predicted_prototype).sum(dim=-1).clamp(0.0, 1.0)

    fused_score = ALPHA * confidence + BETA * topology_score
    return predicted, confidence, topology_score, fused_score


# ============================================================
# 改进方案拓扑打分（中心化 + 相对分数）
# ============================================================
def calculate_topology_scores_improved(data, probs):
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1
    neighbor_dist = _build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes, device=probs.device)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(
                torch.tensor(c, device=probs.device),
                num_classes=num_classes
            ).float()

    # 中心化
    p_bar = prototypes.mean(dim=0, keepdim=True)
    prototypes_tilde = prototypes - p_bar

    # 相对分数
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
# SLGAT 训练（vanilla / old / improved）
# ============================================================
def train_slgat(data, seed, mode, hidden, label_hidden, threshold=0.8):
    set_seed(seed)
    num_classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, num_classes,
                  DROPOUT, label_hidden_dim=label_hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)

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

            if mode == "vanilla":
                pseudo_labels = probs.argmax(dim=-1)
                selected_mask = unlabeled_pool
            elif mode == "old":
                pseudo_labels, _, _, fused_score = calculate_topology_scores_old(data, probs)
                selected_mask = unlabeled_pool & (fused_score >= threshold)
            elif mode == "improved":
                pseudo_labels, _, _, fused_score = calculate_topology_scores_improved(data, probs)
                selected_mask = unlabeled_pool & (fused_score >= threshold)
            else:
                raise ValueError("unknown mode: %s" % mode)

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
# 单个数据集的完整实验
# ============================================================
def run_one_dataset(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    hidden = cfg["hidden"]
    label_hidden = cfg["label_hidden"]
    heads = cfg["heads"]

    out_dir = os.path.join(ROOT_OUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print("# DATASET:", dataset_name)
    print("# config: hidden=%d, label_hidden=%d, heads=%d" % (hidden, label_hidden, heads))
    print("#" * 80)

    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0].to(DEVICE)

    # 打印图统计
    deg = torch.zeros(data.num_nodes, device=DEVICE)
    deg.scatter_add_(0, data.edge_index[1],
                     torch.ones_like(data.edge_index[1], dtype=torch.float))
    n_isolated = int((deg <= 1).sum().item())
    print("nodes=%d, edges=%d, features=%d, classes=%d" % (
        data.num_nodes, data.num_edges, data.num_features,
        int(data.y.max().item()) + 1))
    print("train=%d, val=%d, test=%d, isolated(deg<=1)=%d (%.1f%%)" % (
        int(data.train_mask.sum()), int(data.val_mask.sum()),
        int(data.test_mask.sum()), n_isolated,
        100.0 * n_isolated / data.num_nodes))

    methods = [
        ("Vanilla GAT",            "gat",       None),
        ("SLGAT (vanilla)",        "vanilla",   None),
        ("SLGAT + old topology",   "old",       THRESHOLD),
        ("SLGAT + improved",       "improved",  THRESHOLD),
    ]

    results = []
    t_dataset_start = time.time()

    for seed in SEEDS:
        print("\n--- [%s] seed=%d ---" % (dataset_name, seed))
        for name, mode, thr in methods:
            t0 = time.time()
            if mode == "gat":
                val_acc, test_acc = train_vanilla_gat(data, seed, hidden, heads)
                sel_str = "-"
            else:
                val_acc, test_acc, sel = train_slgat(
                    data, seed, mode=mode,
                    hidden=hidden, label_hidden=label_hidden,
                    threshold=(thr if thr is not None else 0.0)
                )
                sel_str = str(sel)
            dt = time.time() - t0
            print("  %-24s | Val=%.4f | Test=%.4f | Sel=%-6s | %.1fs" % (
                name, val_acc, test_acc, sel_str, dt))
            results.append({
                "seed": seed,
                "method": name,
                "mode": mode,
                "threshold": (thr if thr is not None else ""),
                "val_acc": val_acc,
                "test_acc": test_acc,
                "selected_pseudo_nodes": sel_str,
            })

    total_min = (time.time() - t_dataset_start) / 60.0
    print("\n[%s] total time: %.1f min" % (dataset_name, total_min))

    # 保存逐次结果
    per_run_path = os.path.join(out_dir, "results_per_run.csv")
    with open(per_run_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "method", "mode", "threshold",
            "val_acc", "test_acc", "selected_pseudo_nodes"
        ])
        writer.writeheader()
        writer.writerows(results)

    # 汇总
    print("\n" + "=" * 80)
    print("SUMMARY [%s] (%d seeds)" % (dataset_name, len(SEEDS)))
    print("=" * 80)

    summary_rows = []
    for name, mode, thr in methods:
        group = [r for r in results if r["method"] == name]
        val_values = np.array([r["val_acc"] for r in group])
        test_values = np.array([r["test_acc"] for r in group])
        print("\n%s" % name)
        print("  Val  acc: %.4f +/- %.4f" % (val_values.mean(), val_values.std(ddof=1)))
        print("  Test acc: %.4f +/- %.4f" % (test_values.mean(), test_values.std(ddof=1)))
        for r in group:
            print("    seed=%4d: val=%.4f test=%.4f sel=%s" % (
                r["seed"], r["val_acc"], r["test_acc"], r["selected_pseudo_nodes"]))
        summary_rows.append({
            "dataset": dataset_name,
            "method": name,
            "mode": mode,
            "threshold": (thr if thr is not None else ""),
            "val_mean": val_values.mean(),
            "val_std": val_values.std(ddof=1),
            "test_mean": test_values.mean(),
            "test_std": test_values.std(ddof=1),
        })

    summary_path = os.path.join(out_dir, "results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "mode", "threshold",
            "val_mean", "val_std", "test_mean", "test_std"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n[%s] per-run saved to: %s" % (dataset_name, per_run_path))
    print("[%s] summary saved to: %s" % (dataset_name, summary_path))

    return summary_rows


# ============================================================
# 主程序：三个数据集顺序跑
# ============================================================
def main():
    print("=" * 80)
    print("Full comparison on Cora / Citeseer / Pubmed")
    print("Methods: Vanilla GAT, SLGAT vanilla, SLGAT+old, SLGAT+improved")
    print("Seeds:", SEEDS)
    print("Threshold (old & improved):", THRESHOLD)
    print("device:", DEVICE)
    print("=" * 80)

    t_all_start = time.time()
    all_summary = []
    for ds in DATASETS:
        rows = run_one_dataset(ds)
        all_summary.extend(rows)

    # 跨数据集总汇总
    all_summary_path = os.path.join(ROOT_OUT_DIR, "all_datasets_summary.csv")
    with open(all_summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "mode", "threshold",
            "val_mean", "val_std", "test_mean", "test_std"
        ])
        writer.writeheader()
        writer.writerows(all_summary)

    total_min = (time.time() - t_all_start) / 60.0
    print("\n" + "=" * 80)
    print("ALL DONE. Total time: %.1f min" % total_min)
    print("All summary saved to:", all_summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
