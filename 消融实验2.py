
# -*- coding: utf-8 -*-
"""
TA-SLGAT 四组消融实验
实验组：
1. SLGAT       : 不增加置信度/拓扑筛选，使用所有非训练节点
2. SLGAT-Conf  : 仅使用置信度筛选
3. SLGAT-Topo  : 仅使用结构拓扑得分筛选
4. TA-SLGAT    : 置信度与结构拓扑得分融合筛选

特点：
- 只读取本地 processed/data.pt
- 不调用 Planetoid，不联网，不下载数据
- 不修改原始数据文件
- Cora、CiteSeer、PubMed
- 10 个固定随机种子
- 保存逐种子结果、汇总结果
"""

import os

THREADS = min(2, os.cpu_count() or 1)
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import csv
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, scatter


# ============================================================
# 1. 实验配置
# ============================================================

DATASETS = ["Cora", "CiteSeer", "PubMed"]

SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]

HIDDEN = {
    "Cora": 32,
    "CiteSeer": 32,
    "PubMed": 16,
}

LABEL_HIDDEN = {
    "Cora": 16,
    "CiteSeer": 16,
    "PubMed": 8,
}

DROPOUT = 0.5
LR = 0.005
WEIGHT_DECAY = 5e-4

PRETRAIN_EPOCHS = 200
SELF_TRAIN_EPOCHS = 600
PATIENCE = 100
VALIDATE_EVERY = 5

ALPHA = 0.5
BETA = 0.5
THRESHOLD = 0.9
TEMPERATURE = 0.1
LAMBDA_U = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 优先读取当前工作目录下的 data 文件夹；
# 否则读取用户已存在的 Windows 本地数据目录。
POSSIBLE_ROOTS = [
    Path.cwd() / "data",
    Path(r"D:\dev\vscode机械学习\GNN\data"),
]

OUT_DIR = Path.cwd() / "ta_slgat_ablation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# 2. 通用工具
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_state(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def accuracy(logits, labels, mask):
    if int(mask.sum().item()) == 0:
        return 0.0

    pred = logits[mask].argmax(dim=-1)
    return float(
        (pred == labels[mask]).float().mean().item()
    )


def macro_f1(logits, labels, mask, num_classes):
    pred = logits[mask].argmax(dim=-1).detach().cpu()
    true = labels[mask].detach().cpu()

    values = []

    for c in range(num_classes):
        tp = int(((pred == c) & (true == c)).sum().item())
        fp = int(((pred == c) & (true != c)).sum().item())
        fn = int(((pred != c) & (true == c)).sum().item())

        denominator = 2 * tp + fp + fn

        if denominator == 0:
            values.append(0.0)
        else:
            values.append(2.0 * tp / denominator)

    return float(np.mean(values))


def should_validate(epoch, max_epochs):
    return (
        epoch == 0
        or (epoch + 1) % VALIDATE_EVERY == 0
        or epoch + 1 == max_epochs
    )


def sharpen_probabilities(probs):
    probs = probs.clamp_min(1e-12)
    return F.softmax(
        torch.log(probs) / TEMPERATURE,
        dim=-1
    )


# ============================================================
# 3. 只加载本地数据，不调用 Planetoid
# ============================================================

def find_local_root():
    for root in POSSIBLE_ROOTS:
        all_exist = all(
            (root / name / "processed" / "data.pt").is_file()
            for name in DATASETS
        )

        if all_exist:
            return root

    expected = "\n".join(
        str(root / name / "processed" / "data.pt")
        for root in POSSIBLE_ROOTS
        for name in DATASETS
    )

    raise FileNotFoundError(
        "没有找到三个数据集的本地 processed/data.pt。\n"
        "程序已停止，不会尝试联网或下载。\n"
        + expected
    )


def extract_data_object(obj):
    """
    兼容常见 PyG processed/data.pt 格式：
    1. Data 对象
    2. (Data, slices) 元组
    3. 包含 Data 的字典
    """

    if isinstance(obj, Data):
        return obj

    if isinstance(obj, (tuple, list)):
        for item in obj:
            if isinstance(item, Data):
                return item

        for item in obj:
            if isinstance(item, dict):
                result = extract_data_object(item)
                if result is not None:
                    return result

    if isinstance(obj, dict):
        for key in ("data", "graph", "dataset"):
            if key in obj:
                result = extract_data_object(obj[key])
                if result is not None:
                    return result

        required = {
            "x", "edge_index", "y",
            "train_mask", "val_mask", "test_mask"
        }

        if required.issubset(obj.keys()):
            return Data(**{
                key: obj[key]
                for key in required
            })

    return None


def load_local_data(root, name):
    path = root / name / "processed" / "data.pt"

    if not path.is_file():
        raise FileNotFoundError(
            f"本地数据文件不存在：{path}"
        )

    # 仅读取本地文件。兼容不同 PyTorch 版本。
    try:
        saved_obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        saved_obj = torch.load(
            path,
            map_location="cpu"
        )

    data = extract_data_object(saved_obj)

    if data is None:
        raise RuntimeError(
            f"无法从 {path} 中识别 PyG Data 对象。"
            "程序已停止，没有修改该文件。"
        )

    required_fields = [
        "x", "edge_index", "y",
        "train_mask", "val_mask", "test_mask"
    ]

    for field in required_fields:
        if getattr(data, field, None) is None:
            raise RuntimeError(
                f"{name} 的本地数据缺少字段：{field}"
            )

    # 转换到 CPU/GPU 设备，仅改变内存中的对象。
    data = data.to(DEVICE)

    data.x = data.x.float()
    data.y = data.y.long()
    data.edge_index = data.edge_index.long()

    data.train_mask = data.train_mask.bool()
    data.val_mask = data.val_mask.bool()
    data.test_mask = data.test_mask.bool()

    return data


def prepare_graph(data):
    num_nodes = data.x.size(0)
    classes = int(data.y.max().item()) + 1

    # 在内存中为每个节点增加自环，用于构造邻居标签分布。
    # 不会写回或修改原始文件。
    node_ids = torch.arange(
        num_nodes,
        device=data.edge_index.device
    )

    data.src_all = torch.cat(
        [data.edge_index[0], node_ids]
    )
    data.dst_all = torch.cat(
        [data.edge_index[1], node_ids]
    )

    data.topology_counts = scatter(
        torch.ones(
            data.dst_all.numel(),
            device=data.x.device,
            dtype=data.x.dtype
        ),
        data.dst_all,
        dim=0,
        dim_size=num_nodes,
        reduce="sum"
    ).clamp_min(1.0)

    data.onehot = F.one_hot(
        data.y,
        num_classes=classes
    ).to(data.x.dtype)

    return data


# ============================================================
# 4. SLGAT 模型
# 沿用之前四模型主实验脚本中的模型结构
# ============================================================

class LabelGuidedAttention(nn.Module):
    def __init__(self, in_dim, label_dim, out_dim, dropout):
        super().__init__()

        self.dropout = dropout

        self.x_proj = nn.Linear(
            in_dim, out_dim, bias=False
        )
        self.g_proj = nn.Linear(
            label_dim, out_dim, bias=False
        )

        self.att = nn.Parameter(
            torch.empty(2 * out_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_dim)
        )

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))

    def forward(self, x, label_repr, edge_index):
        src, dst = edge_index

        h = self.x_proj(x)
        g = self.g_proj(label_repr)

        score = (
            g[dst] * self.att[:g.size(-1)]
        ).sum(dim=-1)

        score = score + (
            h[src] * self.att[g.size(-1):]
        ).sum(dim=-1)

        score = F.leaky_relu(score, negative_slope=0.2)

        alpha = softmax(
            score,
            dst,
            num_nodes=x.size(0)
        )

        alpha = F.dropout(
            alpha,
            p=self.dropout,
            training=self.training
        )

        messages = alpha.unsqueeze(-1) * h[src]

        out = scatter(
            messages,
            dst,
            dim=0,
            dim_size=x.size(0),
            reduce="sum"
        )

        return out + self.bias


class SLGAT(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        label_hidden,
        num_classes,
        dropout
    ):
        super().__init__()

        self.dropout = dropout

        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, num_classes)

        self.lc1 = GCNConv(
            num_classes, label_hidden
        )
        self.lc2 = GCNConv(
            label_hidden, label_hidden
        )

        self.att1 = LabelGuidedAttention(
            in_dim, label_hidden, hidden, dropout
        )
        self.att2 = LabelGuidedAttention(
            hidden, label_hidden, num_classes, dropout
        )

    def forward(self, x, edge_index):
        # GCN 分类分支
        z = F.dropout(
            x, p=self.dropout, training=self.training
        )
        z = F.relu(
            self.gcn1(z, edge_index)
        )

        z = F.dropout(
            z, p=self.dropout, training=self.training
        )
        gcn_logits = self.gcn2(z, edge_index)

        # 软标签传播分支
        label_repr = F.relu(
            self.lc1(
                F.softmax(gcn_logits, dim=-1),
                edge_index
            )
        )

        label_repr = F.dropout(
            label_repr,
            p=self.dropout,
            training=self.training
        )

        label_repr = self.lc2(
            label_repr, edge_index
        )

        # 标签引导注意力分支
        h = F.dropout(
            x, p=self.dropout, training=self.training
        )
        h = F.elu(
            self.att1(h, label_repr, edge_index)
        )

        h = F.dropout(
            h, p=self.dropout, training=self.training
        )

        att_logits = self.att2(
            h, label_repr, edge_index
        )

        final_logits = 0.5 * (
            gcn_logits + att_logits
        )

        return final_logits, gcn_logits, att_logits


# ============================================================
# 5. 结构拓扑得分
# 沿用之前四模型主实验脚本中的打分定义
# ============================================================

def calculate_scores(data, probs):
    num_nodes = data.x.size(0)
    num_classes = int(data.y.max().item()) + 1

    # 未标记节点采用预测概率；
    # 训练节点采用真实标签，避免训练标签被模型预测替代。
    signal = probs.detach().clone()
    signal[data.train_mask] = data.onehot[data.train_mask]

    neighbor_sum = scatter(
        signal[data.src_all],
        data.dst_all,
        dim=0,
        dim_size=num_nodes,
        reduce="sum"
    )

    neighbor_dist = (
        neighbor_sum
        / data.topology_counts.unsqueeze(-1)
    )

    # 使用训练节点的邻居标签分布构建类别拓扑原型。
    prototypes = torch.zeros(
        num_classes,
        num_classes,
        device=probs.device,
        dtype=probs.dtype
    )

    for c in range(num_classes):
        mask = data.train_mask & (data.y == c)

        if mask.any():
            prototypes[c] = neighbor_dist[mask].mean(dim=0)
        else:
            prototypes[c, c] = 1.0

    prototypes = F.normalize(
        prototypes,
        p=2,
        dim=-1
    )

    node_patterns = F.normalize(
        neighbor_dist,
        p=2,
        dim=-1
    )

    # 每个节点的邻居分布与各类别原型的余弦相似度。
    cosine_all = node_patterns @ prototypes.T

    confidence, predicted = probs.max(dim=-1)

    topology_score = cosine_all.gather(
        1,
        predicted.unsqueeze(1)
    ).squeeze(1)

    topology_score = topology_score.clamp(0.0, 1.0)

    fused_score = (
        ALPHA * confidence
        + BETA * topology_score
    )

    return confidence, topology_score, fused_score


# ============================================================
# 6. 四组伪标签筛选规则
# ============================================================

ABLATIONS = [
    "SLGAT",
    "SLGAT-Conf",
    "SLGAT-Topo",
    "TA-SLGAT",
]


def select_pseudo_nodes(
    data,
    probs,
    confidence,
    topology_score,
    fused_score,
    method
):
    # 训练节点不属于伪标签候选集合。
    unlabeled_mask = ~data.train_mask

    if method == "SLGAT":
        # 基础 SLGAT：所有非训练节点均参与自训练。
        selected = unlabeled_mask

    elif method == "SLGAT-Conf":
        # 只保留预测置信度达到阈值的节点。
        selected = (
            unlabeled_mask
            & (confidence >= THRESHOLD)
        )

    elif method == "SLGAT-Topo":
        # 只保留拓扑得分达到阈值的节点。
        selected = (
            unlabeled_mask
            & (topology_score >= THRESHOLD)
        )

    elif method == "TA-SLGAT":
        # 两种得分按固定权重融合。
        selected = (
            unlabeled_mask
            & (fused_score >= THRESHOLD)
        )

    else:
        raise ValueError(f"未知消融实验组：{method}")

    return selected


# ============================================================
# 7. 单个消融实验
# ============================================================

def train_one_ablation(
    data,
    seed,
    method,
    hidden,
    label_hidden
):
    set_seed(seed)

    num_classes = int(data.y.max().item()) + 1

    model = SLGAT(
        in_dim=data.x.size(-1),
        hidden=hidden,
        label_hidden=label_hidden,
        num_classes=num_classes,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # 阶段一：监督预训练
    # --------------------------------------------------------
    best_val = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            data.x, data.edge_index
        )[0]

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        if should_validate(epoch, PRETRAIN_EPOCHS):
            model.eval()

            with torch.inference_mode():
                val_logits = model(
                    data.x, data.edge_index
                )[0]

                val = accuracy(
                    val_logits,
                    data.y,
                    data.val_mask
                )

            if val > best_val:
                best_val = val
                best_state = clone_state(model)
                best_epoch = epoch + 1
                wait = 0
            else:
                wait += VALIDATE_EVERY

        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(
            f"{method} 监督预训练未能保存模型参数。"
        )

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 阶段二：自训练
    # --------------------------------------------------------
    best_val = -1.0
    best_state = clone_state(model)
    best_epoch = PRETRAIN_EPOCHS
    wait = 0
    best_selected_count = 0

    for epoch in range(SELF_TRAIN_EPOCHS):
        # 使用当前模型产生伪标签及筛选分数。
        model.eval()

        with torch.no_grad():
            logits_eval = model(
                data.x, data.edge_index
            )[0]

            probs = F.softmax(logits_eval, dim=-1)

            confidence, topology_score, fused_score = (
                calculate_scores(data, probs)
            )

            selected_mask = select_pseudo_nodes(
                data=data,
                probs=probs,
                confidence=confidence,
                topology_score=topology_score,
                fused_score=fused_score,
                method=method
            )

            selected_count = int(
                selected_mask.sum().item()
            )

            # 与原实验一致：使用温度参数锐化软标签。
            soft_targets = sharpen_probabilities(
                probs
            ).detach()

        # 监督损失 + 伪标签无监督损失
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(
            data.x, data.edge_index
        )[0]

        loss_sup = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        )

        if selected_mask.any():
            log_probs = F.log_softmax(
                logits[selected_mask],
                dim=-1
            )

            loss_unsup = -(
                soft_targets[selected_mask] * log_probs
            ).sum(dim=-1).mean()
        else:
            loss_unsup = logits.sum() * 0.0

        loss = loss_sup + LAMBDA_U * loss_unsup

        loss.backward()
        optimizer.step()

        # 验证集早停
        if should_validate(epoch, SELF_TRAIN_EPOCHS):
            model.eval()

            with torch.inference_mode():
                val_logits = model(
                    data.x, data.edge_index
                )[0]

                val = accuracy(
                    val_logits,
                    data.y,
                    data.val_mask
                )

            if val > best_val:
                best_val = val
                best_state = clone_state(model)
                best_epoch = PRETRAIN_EPOCHS + epoch + 1
                best_selected_count = selected_count
                wait = 0
            else:
                wait += VALIDATE_EVERY

        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                f"      {method:10s} "
                f"self-epoch={epoch + 1:3d}, "
                f"best-val={best_val:.4f}, "
                f"selected={selected_count}",
                flush=True
            )

        if wait >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        final_logits = model(
            data.x, data.edge_index
        )[0]

        test_acc = accuracy(
            final_logits,
            data.y,
            data.test_mask
        )

        test_f1 = macro_f1(
            final_logits,
            data.y,
            data.test_mask,
            num_classes
        )

    elapsed = time.perf_counter() - start_time

    return {
        "val_acc": best_val,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "selected_pseudo_nodes": best_selected_count,
    }


# ============================================================
# 8. 结果保存
# ============================================================

PER_SEED_FIELDS = [
    "dataset",
    "seed",
    "model",
    "val_acc",
    "test_acc",
    "test_macro_f1",
    "best_epoch",
    "elapsed_sec",
    "selected_pseudo_nodes",
]


def save_per_seed(rows):
    path = OUT_DIR / "ablation_results_per_seed.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=PER_SEED_FIELDS
        )
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows):
    summary = []

    metrics = [
        "val_acc",
        "test_acc",
        "test_macro_f1",
        "elapsed_sec",
        "selected_pseudo_nodes",
    ]

    print("\n" + "=" * 92)
    print("消融实验汇总：均值 ± 样本标准差")
    print("=" * 92)

    for dataset_name in DATASETS:
        for method in ABLATIONS:
            group = [
                row for row in rows
                if row["dataset"] == dataset_name
                and row["model"] == method
            ]

            if not group:
                continue

            item = {
                "dataset": dataset_name,
                "model": method,
                "n_seeds": len(group),
            }

            for metric in metrics:
                values = np.asarray(
                    [float(row[metric]) for row in group],
                    dtype=float
                )

                item[metric + "_mean"] = float(values.mean())

                item[metric + "_std"] = (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0
                )

            summary.append(item)

            print(
                f"{dataset_name:8s} {method:10s} "
                f"Acc={item['test_acc_mean']:.4f}"
                f"±{item['test_acc_std']:.4f}  "
                f"Macro-F1={item['test_macro_f1_mean']:.4f}"
                f"±{item['test_macro_f1_std']:.4f}  "
                f"Pseudo={item['selected_pseudo_nodes_mean']:.1f}"
            )

    summary_path = OUT_DIR / "ablation_summary_mean_std.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n逐种子结果：{(OUT_DIR / 'ablation_results_per_seed.csv').resolve()}")
    print(f"汇总结果：{summary_path.resolve()}")


# ============================================================
# 9. 主程序
# ============================================================

def main():
    print("=" * 92)
    print("TA-SLGAT 四组消融实验")
    print("数据集：Cora / CiteSeer / PubMed")
    print("实验组：SLGAT / SLGAT-Conf / SLGAT-Topo / TA-SLGAT")
    print(f"随机种子：{SEEDS}")
    print(f"设备：{DEVICE}")
    print(f"线程数：{THREADS}")
    print(f"阈值：{THRESHOLD}")
    print(f"融合权重：alpha={ALPHA}, beta={BETA}")
    print(f"输出目录：{OUT_DIR.resolve()}")
    print("=" * 92)

    root = find_local_root()
    print(f"本地数据根目录：{root}")
    print("只读取本地文件，不联网、不下载。")

    all_rows = []

    for dataset_name in DATASETS:
        data = prepare_graph(
            load_local_data(root, dataset_name)
        )

        hidden = HIDDEN[dataset_name]
        label_hidden = LABEL_HIDDEN[dataset_name]
        num_classes = int(data.y.max().item()) + 1

        print("\n" + "#" * 92)
        print(
            f"{dataset_name}: "
            f"nodes={data.x.size(0)}, "
            f"edges={data.edge_index.size(1)}, "
            f"features={data.x.size(1)}, "
            f"classes={num_classes}, "
            f"train/val/test="
            f"{int(data.train_mask.sum())}/"
            f"{int(data.val_mask.sum())}/"
            f"{int(data.test_mask.sum())}"
        )
        print("#" * 92)

        for seed in SEEDS:
            print(f"\n--- {dataset_name} seed={seed} ---")

            for method in ABLATIONS:
                result = train_one_ablation(
                    data=data,
                    seed=seed,
                    method=method,
                    hidden=hidden,
                    label_hidden=label_hidden
                )

                row = {
                    "dataset": dataset_name,
                    "seed": seed,
                    "model": method,
                    **result,
                }

                all_rows.append(row)

                print(
                    f"{method:10s} "
                    f"val={result['val_acc']:.4f} "
                    f"test_acc={result['test_acc']:.4f} "
                    f"Macro-F1={result['test_macro_f1']:.4f} "
                    f"pseudo_nodes={result['selected_pseudo_nodes']} "
                    f"best_epoch={result['best_epoch']} "
                    f"time={result['elapsed_sec']:.1f}s",
                    flush=True
                )

            # 每完成一个随机种子就保存，降低中断造成的数据损失。
            save_per_seed(all_rows)

    save_summary(all_rows)

    print("\n全部消融实验完成。")


if __name__ == "__main__":
    main()
