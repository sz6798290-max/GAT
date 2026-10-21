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
# 1. 实验配置
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT_OUT_DIR = os.path.join(
    os.getcwd(), "slgat_ablation"
)
os.makedirs(ROOT_OUT_DIR, exist_ok=True)


# ============================================================
# 2. 消融配置
# ============================================================
ABLATION_CONFIGS = {
    "full": {
        "use_topology": True,
        "use_centering": True,
        "use_relative": True,
    },
    "confidence_only": {
        "use_topology": False,
        "use_centering": True,   # 不生效
        "use_relative": True,    # 不生效
    },
    "wo_centering": {
        "use_topology": True,
        "use_centering": False,
        "use_relative": True,
    },
    "absolute_sim": {
        "use_topology": True,
        "use_centering": True,
        "use_relative": False,
    },
    "wo_both": {
        "use_topology": True,
        "use_centering": False,
        "use_relative": False,
    },
}

# 想只跑某几组就改这里
RUN_ABLATIONS = [
    "full",
    "confidence_only",
    "wo_centering",
    "absolute_sim",
    "wo_both",
]

CURRENT_ABLATION = ABLATION_CONFIGS["full"]
ABLATION_NAME = "full"


# ============================================================
# 3. 工具函数
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
    if int(mask.sum().item()) == 0:
        return 0.0

    pred = logits[mask].argmax(dim=-1)
    return (pred == labels[mask]).float().mean().item()


def supervised_loss(logits, labels, mask):
    return F.cross_entropy(logits[mask], labels[mask])


def sharpen_probabilities(
    probs, temperature=PSEUDO_TEMPERATURE
):
    probs = probs.clamp_min(1e-12)
    return F.softmax(torch.log(probs) / temperature, dim=-1)


# ============================================================
# 4. 标签引导注意力层
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
        e = torch.tanh((pair * self.att).sum(dim=-1))

        alpha = softmax(e, dst, num_nodes=x.size(0))

        alpha = F.dropout(
            alpha, p=self.dropout, training=self.training
        )

        messages = alpha.unsqueeze(-1) * h[src]

        out = scatter(
            messages, dst, dim=0,
            dim_size=x.size(0), reduce="sum"
        )

        return out + self.bias


# ============================================================
# 5. SLGAT 模型
# ============================================================
class SLGAT(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, num_classes,
        dropout, label_hidden_dim=16
    ):
        super().__init__()

        self.dropout = dropout

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, num_classes)

        self.label_conv1 = GCNConv(
            num_classes, label_hidden_dim
        )
        self.label_conv2 = GCNConv(
            label_hidden_dim, label_hidden_dim
        )

        self.att1 = LabelGuidedAttention(
            in_dim, label_hidden_dim, hidden_dim, dropout
        )
        self.att2 = LabelGuidedAttention(
            hidden_dim, label_hidden_dim, num_classes, dropout
        )

    def forward(self, x, edge_index):
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.relu(self.gcn1(z, edge_index))

        z = F.dropout(z, p=self.dropout, training=self.training)
        gcn_logits = self.gcn2(z, edge_index)

        soft_labels = F.softmax(gcn_logits, dim=-1)

        label_repr = F.relu(
            self.label_conv1(soft_labels, edge_index)
        )
        label_repr = F.dropout(
            label_repr, p=self.dropout, training=self.training
        )
        label_repr = self.label_conv2(label_repr, edge_index)

        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.att1(h, label_repr, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        att_logits = self.att2(h, label_repr, edge_index)

        final_logits = 0.5 * (gcn_logits + att_logits)

        return final_logits, gcn_logits, att_logits


# ============================================================
# 6. 构造邻居标签分布
# ============================================================
def _build_neighbor_dist(data, probs, num_classes):
    num_nodes = data.num_nodes
    src, dst = data.edge_index

    label_signal = probs.detach().clone()

    true_onehot = F.one_hot(
        data.y, num_classes=num_classes
    ).to(dtype=probs.dtype)

    label_signal[data.train_mask] = (
        true_onehot[data.train_mask]
    )

    node_ids = torch.arange(
        num_nodes, device=probs.device
    )

    src_all = torch.cat([src, node_ids])
    dst_all = torch.cat([dst, node_ids])

    sums = scatter(
        label_signal[src_all], dst_all,
        dim=0, dim_size=num_nodes, reduce="sum"
    )

    counts = scatter(
        torch.ones(
            dst_all.numel(),
            device=probs.device,
            dtype=probs.dtype
        ),
        dst_all, dim=0, dim_size=num_nodes, reduce="sum"
    ).clamp_min(1.0)

    neighbor_dist = sums / counts.unsqueeze(-1)

    return neighbor_dist


# ============================================================
# 7. 拓扑感知打分（带消融开关）
# ============================================================
def calculate_topology_scores_improved(data, probs):
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1

    neighbor_dist = _build_neighbor_dist(
        data, probs, num_classes
    )

    prototypes = torch.zeros(
        num_classes, num_classes,
        device=probs.device, dtype=probs.dtype
    )

    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)

        if class_mask.any():
            prototypes[c] = neighbor_dist[
                class_mask
            ].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(
                torch.tensor(c, device=probs.device),
                num_classes=num_classes
            ).to(dtype=probs.dtype)

    # ---- 消融：是否做类别原型中心化 ----
    if CURRENT_ABLATION["use_centering"]:
        p_bar = prototypes.mean(dim=0, keepdim=True)
        prototypes_tilde = prototypes - p_bar
    else:
        prototypes_tilde = prototypes

    node_pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    proto_norm = F.normalize(prototypes_tilde, p=2, dim=-1)

    cos_all = node_pattern @ proto_norm.T

    confidence, predicted = probs.max(dim=-1)

    pred_cos = cos_all.gather(
        1, predicted.unsqueeze(1)
    ).squeeze(1)

    mask_other = torch.ones_like(
        cos_all, dtype=torch.bool
    )
    mask_other.scatter_(
        1, predicted.unsqueeze(1), False
    )

    other_max = cos_all.masked_fill(
        ~mask_other, -1e9
    ).max(dim=1).values

    # ---- 消融：相对相似度 vs 绝对相似度 ----
    if CURRENT_ABLATION["use_relative"]:
        topology_score = (
            pred_cos - other_max + 2.0
        ) / 4.0
    else:
        topology_score = (pred_cos + 1.0) / 2.0

    topology_score = topology_score.clamp(0.0, 1.0)

    # ---- 消融：是否使用拓扑分数 ----
    if CURRENT_ABLATION["use_topology"]:
        fused_score = ALPHA * confidence + BETA * topology_score
    else:
        fused_score = confidence

    return predicted, confidence, topology_score, fused_score


# ============================================================
# 8. 训练 SLGAT + 拓扑感知伪标签筛选
# ============================================================
def train_slgat_improved(
    data, seed, hidden, label_hidden, threshold=0.8
):
    set_seed(seed)

    num_classes = int(data.y.max().item()) + 1

    model = SLGAT(
        data.num_features,
        hidden,
        num_classes,
        DROPOUT,
        label_hidden_dim=label_hidden
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # 阶段一：监督预训练
    # --------------------------------------------------------
    best_val = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(data.x, data.edge_index)

        loss = supervised_loss(
            logits, data.y, data.train_mask
        )

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(data.x, data.edge_index)

            val_acc = accuracy(
                logits, data.y, data.val_mask
            )

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("监督预训练没有生成有效模型状态。")

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 阶段二：自训练
    # --------------------------------------------------------
    best_val = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    selected_count = 0

    unlabeled_pool = ~data.train_mask

    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()

        with torch.no_grad():
            logits, _, _ = model(
                data.x, data.edge_index
            )
            probs = F.softmax(logits, dim=-1)

            _, _, _, fused_score = (
                calculate_topology_scores_improved(
                    data, probs
                )
            )

            selected_mask = (
                unlabeled_pool
                & (fused_score >= threshold)
            )

            selected_count = int(
                selected_mask.sum().item()
            )

            soft_targets = sharpen_probabilities(
                probs
            ).detach()

        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(data.x, data.edge_index)

        loss_sup = supervised_loss(
            logits, data.y, data.train_mask
        )

        if selected_mask.any():
            log_probs = F.log_softmax(
                logits[selected_mask], dim=-1
            )

            loss_unsup = -(
                soft_targets[selected_mask] * log_probs
            ).sum(dim=-1).mean()
        else:
            loss_unsup = logits.sum() * 0.0

        loss = loss_sup + LAMBDA_U * loss_unsup

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _, _ = model(
                data.x, data.edge_index
            )

            val_acc = accuracy(
                logits, data.y, data.val_mask
            )

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

        test_acc = accuracy(
            logits, data.y, data.test_mask
        )

        # ====================================================
        # 用最终模型重新算一次筛选结果，统计伪标签精度
        # ====================================================
        probs = F.softmax(logits, dim=-1)

        (
            predicted,
            confidence,
            topology_score,
            fused_score
        ) = calculate_topology_scores_improved(
            data, probs
        )

        selected_mask = (
            unlabeled_pool
            & (fused_score >= threshold)
        )

        if selected_mask.any():
            pseudo_correct = (
                predicted[selected_mask]
                == data.y[selected_mask]
            ).float()

            pseudo_precision = pseudo_correct.mean().item()

            selected_labels = data.y[selected_mask]
            class_counts = torch.bincount(
                selected_labels,
                minlength=num_classes
            ).float()

            class_ratio = (
                class_counts / class_counts.sum()
            )

            class_balance_cv = (
                class_ratio.std() / class_ratio.mean()
            ).item() if class_ratio.mean() > 0 else 0.0
        else:
            pseudo_precision = 0.0
            class_balance_cv = 0.0

    return (
        best_val,
        test_acc,
        selected_count,
        pseudo_precision,
        class_balance_cv
    )


# ============================================================
# 9. 单个数据集 + 单个消融的完整实验
# ============================================================
def run_one_dataset(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]

    hidden = cfg["hidden"]
    label_hidden = cfg["label_hidden"]

    out_dir = os.path.join(
        ROOT_OUT_DIR, ABLATION_NAME, dataset_name
    )
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print("# ABLATION:", ABLATION_NAME)
    print("# DATASET:", dataset_name)
    print(
        "# hidden=%d, label_hidden=%d"
        % (hidden, label_hidden)
    )
    print("# device:", DEVICE)
    print("#" * 80)

    dataset = Planetoid(
        root="./data", name=dataset_name
    )
    data = dataset[0].to(DEVICE)

    results = []
    t_dataset_start = time.time()

    for seed in SEEDS:
        print(
            "\n--- [%s | %s] seed=%d ---"
            % (ABLATION_NAME, dataset_name, seed)
        )

        t0 = time.time()

        (
            val_acc,
            test_acc,
            selected_count,
            pseudo_precision,
            class_balance_cv
        ) = train_slgat_improved(
            data,
            seed,
            hidden=hidden,
            label_hidden=label_hidden,
            threshold=THRESHOLD
        )

        elapsed = time.time() - t0

        print(
            "Ablation=%s | "
            "Val=%.4f | Test=%.4f | "
            "Selected=%d | PseudoPrec=%.4f | "
            "ClassCV=%.4f | Time=%.1fs"
            % (
                ABLATION_NAME,
                val_acc,
                test_acc,
                selected_count,
                pseudo_precision,
                class_balance_cv,
                elapsed
            )
        )

        results.append({
            "seed": seed,
            "ablation": ABLATION_NAME,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "selected_pseudo_nodes": selected_count,
            "pseudo_precision": pseudo_precision,
            "class_balance_cv": class_balance_cv,
            "time_seconds": elapsed
        })

    total_min = (
        time.time() - t_dataset_start
    ) / 60.0

    per_run_path = os.path.join(
        out_dir, "results_per_run.csv"
    )

    with open(
        per_run_path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "ablation",
                "val_acc",
                "test_acc",
                "selected_pseudo_nodes",
                "pseudo_precision",
                "class_balance_cv",
                "time_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    val_values = np.array(
        [r["val_acc"] for r in results]
    )
    test_values = np.array(
        [r["test_acc"] for r in results]
    )
    selected_values = np.array(
        [r["selected_pseudo_nodes"] for r in results]
    )
    pseudo_values = np.array(
        [r["pseudo_precision"] for r in results]
    )
    cv_values = np.array(
        [r["class_balance_cv"] for r in results]
    )

    val_mean = val_values.mean()
    val_std = val_values.std(ddof=1)
    test_mean = test_values.mean()
    test_std = test_values.std(ddof=1)
    selected_mean = selected_values.mean()
    pseudo_mean = pseudo_values.mean()
    pseudo_std = pseudo_values.std(ddof=1)
    cv_mean = cv_values.mean()

    print("\n" + "=" * 80)
    print(
        "SUMMARY [%s | %s] (%d seeds)"
        % (ABLATION_NAME, dataset_name, len(SEEDS))
    )
    print("=" * 80)
    print(
        "Validation accuracy: %.4f +/- %.4f"
        % (val_mean, val_std)
    )
    print(
        "Test accuracy:       %.4f +/- %.4f"
        % (test_mean, test_std)
    )
    print(
        "Selected pseudo nodes (mean): %.1f"
        % selected_mean
    )
    print(
        "Pseudo-label precision: %.4f +/- %.4f"
        % (pseudo_mean, pseudo_std)
    )
    print(
        "Class balance CV (mean): %.4f"
        % cv_mean
    )
    print("Total time: %.1f min" % total_min)

    summary_row = {
        "ablation": ABLATION_NAME,
        "dataset": dataset_name,
        "val_mean": val_mean,
        "val_std": val_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "selected_mean": selected_mean,
        "pseudo_precision_mean": pseudo_mean,
        "pseudo_precision_std": pseudo_std,
        "class_balance_cv_mean": cv_mean
    }

    summary_path = os.path.join(
        out_dir, "results_summary.csv"
    )

    with open(
        summary_path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation",
                "dataset",
                "val_mean",
                "val_std",
                "test_mean",
                "test_std",
                "selected_mean",
                "pseudo_precision_mean",
                "pseudo_precision_std",
                "class_balance_cv_mean"
            ]
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print("Per-run results saved to:", per_run_path)
    print("Summary saved to:", summary_path)

    return summary_row


# ============================================================
# 10. 主程序
# ============================================================
def main():
    global CURRENT_ABLATION, ABLATION_NAME

    print("=" * 80)
    print("SLGAT Ablation Study")
    print("Ablations:", RUN_ABLATIONS)
    print("Datasets:", DATASETS)
    print("Seeds:", SEEDS)
    print("Device:", DEVICE)
    print("=" * 80)

    t_all_start = time.time()
    all_summary = []

    for ab_name in RUN_ABLATIONS:
        cfg = ABLATION_CONFIGS[ab_name]
        CURRENT_ABLATION = cfg
        ABLATION_NAME = ab_name

        print("\n\n" + "%" * 80)
        print("%% START ABLATION:", ab_name)
        print(
            "%% Config: use_topology=%s, "
            "use_centering=%s, use_relative=%s"
            % (
                cfg["use_topology"],
                cfg["use_centering"],
                cfg["use_relative"]
            )
        )
        print("%" * 80)

        ab_summary = []

        for dataset_name in DATASETS:
            summary_row = run_one_dataset(dataset_name)
            ab_summary.append(summary_row)
            all_summary.append(summary_row)

        ab_summary_path = os.path.join(
            ROOT_OUT_DIR, ab_name,
            "all_datasets_summary.csv"
        )

        with open(
            ab_summary_path,
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ablation",
                    "dataset",
                    "val_mean",
                    "val_std",
                    "test_mean",
                    "test_std",
                    "selected_mean",
                    "pseudo_precision_mean",
                    "pseudo_precision_std",
                    "class_balance_cv_mean"
                ]
            )
            writer.writeheader()
            writer.writerows(ab_summary)

    all_summary_path = os.path.join(
        ROOT_OUT_DIR, "ablation_all_datasets.csv"
    )

    with open(
        all_summary_path,
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation",
                "dataset",
                "val_mean",
                "val_std",
                "test_mean",
                "test_std",
                "selected_mean",
                "pseudo_precision_mean",
                "pseudo_precision_std",
                "class_balance_cv_mean"
            ]
        )
        writer.writeheader()
        writer.writerows(all_summary)

    total_min = (
        time.time() - t_all_start
    ) / 60.0

    print("\n" + "=" * 80)
    print("ALL ABLATION EXPERIMENTS FINISHED")
    print("Total time: %.1f min" % total_min)
    print("All results saved to:", all_summary_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
