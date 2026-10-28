# -*- coding: utf-8 -*-
"""随机划分鲁棒性实验：GCN / GAT / SLGAT / TA-SLGAT。
3 数据集 × 3 随机划分 × 3 随机种子 = 108 次训练。
随机划分规则：每类 20 训练节点；剩余随机 500 验证、1000 测试。
"""
import os
THREADS = int(os.environ.get("GNN_CPU_THREADS", str(min(2, os.cpu_count() or 1))))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import csv, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import softmax, scatter

# ====== 随机划分实验配置 ======
DATASETS = ["Cora", "CiteSeer", "PubMed"]
SPLITS = [1, 2, 3]          # 3 个随机数据划分
SEEDS = [1, 2, 3]           # 每个划分 3 个随机种子
TRAIN_PER_CLASS = 20        # 每类训练节点数（与标准划分一致）
VAL_SIZE, TEST_SIZE = 500, 1000
# =============================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
ALPHA, BETA, THRESHOLD, TEMPERATURE, LAMBDA_U = 0.5, 0.5, 0.9, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_random_split_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def acc(logits, y, mask):
    if not mask.any(): return 0.0
    return float((logits[mask].argmax(-1) == y[mask]).float().mean().item())


def macro_f1(logits, y, mask, classes):
    pred = logits[mask].argmax(-1).detach().cpu()
    true = y[mask].detach().cpu()
    values = []
    for c in range(classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        den = 2 * tp + fp + fn
        values.append(2 * tp / den if den else 0.0)
    return float(np.mean(values))


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def logits_of(output):
    return output[0] if isinstance(output, tuple) else output


def validate_now(epoch, max_epochs):
    return epoch == 0 or (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == max_epochs


def data_root():
    env_root = os.environ.get("GNN_DATA_ROOT")
    root = Path(env_root) if env_root else (Path.cwd() / "data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_data(root, name):
    ds = Planetoid(root=str(root), name=name)
    data = ds[0].to(DEVICE)
    for field in ("x", "edge_index", "y", "train_mask", "val_mask", "test_mask"):
        if getattr(data, field, None) is None:
            raise RuntimeError(f"{name} 缺少字段：{field}")
    return data


def prepare_graph(data):
    n = data.num_nodes
    ids = torch.arange(n, device=data.edge_index.device)
    data.src_all = torch.cat([data.edge_index[0], ids])
    data.dst_all = torch.cat([data.edge_index[1], ids])
    data.topology_counts = scatter(
        torch.ones(data.dst_all.numel(), device=DEVICE, dtype=data.x.dtype),
        data.dst_all, dim=0, dim_size=n, reduce="sum"
    ).clamp_min(1)
    classes = int(data.y.max().item()) + 1
    data.onehot = F.one_hot(data.y, num_classes=classes).to(data.x.dtype)
    data.class_train_masks = [data.train_mask & (data.y == c) for c in range(classes)]
    return data


# ====== 随机划分生成器 ======
def make_random_split(data, split_id):
    """每类随机 20 个训练；剩余随机 500 验证、1000 测试。
    用 split_id 作为随机种子，保证同一 split 每次生成相同划分。
    """
    rng = np.random.RandomState(split_id)
    n = data.num_nodes
    y = data.y.cpu().numpy()
    classes = int(data.y.max().item()) + 1

    train_idx = []
    for c in range(classes):
        class_nodes = np.where(y == c)[0]
        if len(class_nodes) < TRAIN_PER_CLASS:
            chosen = class_nodes  # 极端情况回退（Planetoid 不会触发）
        else:
            chosen = rng.choice(class_nodes, TRAIN_PER_CLASS, replace=False)
        train_idx.extend(chosen.tolist())
    train_idx = np.array(train_idx, dtype=np.int64)

    # 剩余节点
    mask_all = np.ones(n, dtype=bool)
    mask_all[train_idx] = False
    remaining = np.where(mask_all)[0]
    rng.shuffle(remaining)
    if len(remaining) < VAL_SIZE + TEST_SIZE:
        raise RuntimeError(f"剩余节点不足：{len(remaining)} < {VAL_SIZE + TEST_SIZE}")
    val_idx = remaining[:VAL_SIZE]
    test_idx = remaining[VAL_SIZE:VAL_SIZE + TEST_SIZE]

    train_mask = torch.zeros(n, dtype=torch.bool, device=data.y.device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=data.y.device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=data.y.device)
    train_mask[torch.from_numpy(train_idx).to(data.y.device)] = True
    val_mask[torch.from_numpy(val_idx).to(data.y.device)] = True
    test_mask[torch.from_numpy(test_idx).to(data.y.device)] = True
    return train_mask, val_mask, test_mask


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, classes, dropout):
        super().__init__(); self.dropout = dropout
        self.c1 = GCNConv(in_dim, hidden); self.c2 = GCNConv(hidden, classes)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.c2(x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, classes, dropout):
        super().__init__(); self.dropout = dropout
        self.g1 = GATConv(in_dim, hidden, heads=8, concat=True, dropout=dropout)
        self.g2 = GATConv(hidden * 8, classes, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.g1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.g2(x, edge_index)


class LabelGuidedAttention(nn.Module):
    def __init__(self, in_dim, label_dim, out_dim, dropout):
        super().__init__(); self.dropout = dropout
        self.x_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.g_proj = nn.Linear(label_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(2 * out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.g_proj.weight)
        nn.init.xavier_uniform_(self.att.view(1, -1))
    def forward(self, x, label_repr, edge_index):
        src, dst = edge_index
        h, g = self.x_proj(x), self.g_proj(label_repr)
        score = (g[dst] * self.att[:g.size(-1)]).sum(-1)
        score = score + (h[src] * self.att[g.size(-1):]).sum(-1)
        alpha = softmax(F.leaky_relu(score, 0.2), dst, num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = scatter(alpha.unsqueeze(-1) * h[src], dst, dim=0,
                      dim_size=x.size(0), reduce="sum")
        return out + self.bias


class SLGAT(nn.Module):
    def __init__(self, in_dim, hidden, label_hidden, classes, dropout):
        super().__init__(); self.dropout = dropout
        self.gcn1 = GCNConv(in_dim, hidden); self.gcn2 = GCNConv(hidden, classes)
        self.lc1 = GCNConv(classes, label_hidden)
        self.lc2 = GCNConv(label_hidden, label_hidden)
        self.att1 = LabelGuidedAttention(in_dim, label_hidden, hidden, dropout)
        self.att2 = LabelGuidedAttention(hidden, label_hidden, classes, dropout)
    def forward(self, x, edge_index):
        z = F.dropout(x, p=self.dropout, training=self.training)
        z = F.relu(self.gcn1(z, edge_index))
        z = F.dropout(z, p=self.dropout, training=self.training)
        gcn_logits = self.gcn2(z, edge_index)
        label_repr = F.relu(self.lc1(F.softmax(gcn_logits, -1), edge_index))
        label_repr = F.dropout(label_repr, p=self.dropout, training=self.training)
        label_repr = self.lc2(label_repr, edge_index)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.att1(h, label_repr, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        att_logits = self.att2(h, label_repr, edge_index)
        return 0.5 * (gcn_logits + att_logits), gcn_logits, att_logits


def topology_scores(data, probs):
    classes = data.onehot.size(-1)
    signal = probs.detach().clone()
    signal[data.train_mask] = data.onehot[data.train_mask]
    sums = scatter(signal[data.src_all], data.dst_all, dim=0,
                   dim_size=data.num_nodes, reduce="sum")
    neighbor_dist = sums / data.topology_counts.unsqueeze(-1)
    prototypes = torch.zeros(classes, classes, device=probs.device, dtype=probs.dtype)
    for c, mask in enumerate(data.class_train_masks):
        if mask.any():
            prototypes[c] = neighbor_dist[mask].mean(0)
        else:
            prototypes[c, c] = 1.0
    prototypes = F.normalize(prototypes, p=2, dim=-1)
    confidence, predicted = probs.max(-1)
    pattern = F.normalize(neighbor_dist, p=2, dim=-1)
    topo = (pattern * prototypes[predicted]).sum(-1).clamp(0.0, 1.0)
    return ALPHA * confidence + BETA * topo


def train_supervised(model, data, seed, lr):
    set_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_val, best_state, best_epoch, wait = -1.0, None, 0, 0
    start = time.perf_counter()
    for epoch in range(PRETRAIN_EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = logits_of(model(data.x, data.edge_index))
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        if validate_now(epoch, PRETRAIN_EPOCHS):
            model.eval()
            with torch.inference_mode():
                val = acc(logits_of(model(data.x, data.edge_index)), data.y, data.val_mask)
            if val > best_val:
                best_val, best_state, best_epoch, wait = val, clone_state(model), epoch + 1, 0
            else:
                wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    if best_state is None: raise RuntimeError("监督模型没有保存最佳参数")
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = logits_of(model(data.x, data.edge_index))
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, int(data.y.max()) + 1)
    return dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
                best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
                selected_pseudo_nodes="")


def train_slgat(data, seed, proposed, hidden, label_hidden, lr):
    set_seed(seed); classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    best_val, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(PRETRAIN_EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)[0]
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()
        if validate_now(epoch, PRETRAIN_EPOCHS):
            model.eval()
            with torch.inference_mode():
                val = acc(model(data.x, data.edge_index)[0], data.y, data.val_mask)
            if val > best_val:
                best_val, best_state, best_epoch, wait = val, clone_state(model), epoch + 1, 0
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    if best_state is None: raise RuntimeError("SLGAT 预训练没有保存最佳参数")
    model.load_state_dict(best_state)
    pretrain_actual_epochs = epoch + 1

    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    selected_count, best_selected_count = 0, 0
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_unlabeled = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_unlabeled, -1)
            if proposed:
                score = topology_scores(data, probs)
                selected = (~data.train_mask) & (score >= THRESHOLD)
            else:
                selected = ~data.train_mask
            selected_count = int(selected.sum().item())
            targets = F.softmax(logits_unlabeled / TEMPERATURE, -1)
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)[0]
        loss_sup = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if selected.any():
            loss_unsup = -(targets[selected] * F.log_softmax(logits[selected], -1)).sum(-1).mean()
        else:
            loss_unsup = logits.sum() * 0.0
        (loss_sup + LAMBDA_U * loss_unsup).backward(); opt.step()

        if validate_now(epoch, SELF_TRAIN_EPOCHS):
            model.eval()
            with torch.inference_mode():
                val = acc(model(data.x, data.edge_index)[0], data.y, data.val_mask)
            if val > best_val:
                best_val, best_state, wait = val, clone_state(model), 0
                best_epoch = pretrain_actual_epochs + epoch + 1
                best_selected_count = selected_count
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)
    return dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
                best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
                selected_pseudo_nodes=best_selected_count)


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    print("=" * 92)
    print("Random-Split Robustness | GCN / GAT / SLGAT / TA-SLGAT")
    print(f"Datasets={DATASETS}; splits={SPLITS}; seeds={SEEDS}; "
          f"train/class={TRAIN_PER_CLASS}, val={VAL_SIZE}, test={TEST_SIZE}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 92)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data_base = prepare_graph(load_data(root, name))
        hidden, label_hidden = HIDDEN[name], LABEL_HIDDEN[name]
        lr = LR_MAP[name]
        classes = int(data_base.y.max().item()) + 1
        print(f"\n{'#'*92}\n{name}: nodes={data_base.num_nodes}, "
              f"features={data_base.num_features}, classes={classes}, lr={lr}")

        for split_id in SPLITS:
            # 生成随机划分并覆盖 data 的 mask（不修改图结构相关字段）
            tr, va, te = make_random_split(data_base, split_id)
            data = data_base
            data.train_mask, data.val_mask, data.test_mask = tr, va, te
            # 重算 class_train_masks（TA 打分依赖它）
            data.class_train_masks = [data.train_mask & (data.y == c) for c in range(classes)]
            print(f"\n--- {name} split={split_id} ---")
            print(f"  train/val/test = {int(tr.sum())}/{int(va.sum())}/{int(te.sum())}")

            for seed in SEEDS:
                print(f"  seed={seed}")
                # GCN / GAT
                for model_name, factory in [
                    ("GCN", lambda: GCN(data.num_features, hidden, classes, DROPOUT)),
                    ("GAT", lambda: GAT(data.num_features, hidden, classes, DROPOUT)),
                ]:
                    set_seed(seed)
                    model = factory()
                    r = train_supervised(model, data, seed, lr)
                    r.update(dataset=name, split=split_id, seed=seed, model=model_name)
                    all_rows.append(r)
                    print(f"    {model_name:8s} val={r['val_acc']:.4f} "
                          f"test={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f}")
                # SLGAT / TA-SLGAT
                for model_name, proposed in [("SLGAT", False), ("TA-SLGAT", True)]:
                    r = train_slgat(data, seed, proposed, hidden, label_hidden, lr)
                    r.update(dataset=name, split=split_id, seed=seed, model=model_name)
                    all_rows.append(r)
                    print(f"    {model_name:8s} val={r['val_acc']:.4f} "
                          f"test={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f} "
                          f"pseudo={r['selected_pseudo_nodes']}")

            # 每个 split 跑完就保存一次
            save_csv(
                OUT_DIR / "per_run.csv", all_rows,
                ["dataset", "split", "seed", "model", "val_acc", "test_acc",
                 "test_macro_f1", "best_epoch", "elapsed_sec", "selected_pseudo_nodes"],
            )

    # ====== 汇总 1：每个 (dataset, split, model) 的 3 seed 均值±std ======
    per_split = []
    for name in DATASETS:
        for split_id in SPLITS:
            for model_name in ["GCN", "GAT", "SLGAT", "TA-SLGAT"]:
                g = [r for r in all_rows if r["dataset"] == name
                     and r["split"] == split_id and r["model"] == model_name]
                if not g: continue
                a = np.array([r["test_acc"] for r in g]); f = np.array([r["test_macro_f1"] for r in g])
                per_split.append(dict(
                    dataset=name, split=split_id, model=model_name, n_seeds=len(g),
                    acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                    f1_mean=float(f.mean()), f1_std=float(f.std(ddof=1)) if len(f) > 1 else 0.0,
                ))
    save_csv(OUT_DIR / "per_split.csv", per_split,
             ["dataset", "split", "model", "n_seeds", "acc_mean", "acc_std", "f1_mean", "f1_std"])

    # ====== 汇总 2：9 次（3 split × 3 seed）mean±std ======
    print("\n" + "=" * 92)
    print("SUMMARY over 3 splits x 3 seeds = 9 runs (mean±std)")
    print("=" * 92)
    summary_9 = []
    for name in DATASETS:
        for model_name in ["GCN", "GAT", "SLGAT", "TA-SLGAT"]:
            g = [r for r in all_rows if r["dataset"] == name and r["model"] == model_name]
            if not g: continue
            a = np.array([r["test_acc"] for r in g]); f = np.array([r["test_macro_f1"] for r in g])
            row = dict(dataset=name, model=model_name, n_runs=len(g),
                       acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                       f1_mean=float(f.mean()), f1_std=float(f.std(ddof=1)))
            summary_9.append(row)
            print(f"{name:8s} {model_name:8s} "
                  f"Acc={row['acc_mean']*100:.2f}±{row['acc_std']*100:.2f}  "
                  f"F1={row['f1_mean']*100:.2f}±{row['f1_std']*100:.2f}")
    save_csv(OUT_DIR / "summary_9runs.csv", summary_9,
             ["dataset", "model", "n_runs", "acc_mean", "acc_std", "f1_mean", "f1_std"])

    # ====== 汇总 3：先 split 内平均，再对 3 个 split 求 mean±std（划分敏感性） ======
    print("\n" + "=" * 92)
    print("SUMMARY: split-level mean±std (先 3 seed 平均，再对 3 splits 求 mean±std)")
    print("=" * 92)
    summary_split = []
    for name in DATASETS:
        for model_name in ["GCN", "GAT", "SLGAT", "TA-SLGAT"]:
            split_means = []
            for split_id in SPLITS:
                g = [r for r in all_rows if r["dataset"] == name
                     and r["split"] == split_id and r["model"] == model_name]
                if g:
                    split_means.append(np.mean([r["test_acc"] for r in g]))
            if not split_means: continue
            sm = np.array(split_means)
            row = dict(dataset=name, model=model_name, n_splits=len(sm),
                       acc_mean=float(sm.mean()),
                       acc_std=float(sm.std(ddof=1)) if len(sm) > 1 else 0.0)
            summary_split.append(row)
            print(f"{name:8s} {model_name:8s} "
                  f"Acc={row['acc_mean']*100:.2f}±{row['acc_std']*100:.2f} (split-level)")
    save_csv(OUT_DIR / "summary_split_mean.csv", summary_split,
             ["dataset", "model", "n_splits", "acc_mean", "acc_std"])

    print(f"\n逐次结果：{(OUT_DIR / 'per_run.csv').resolve()}")
    print(f"逐划分汇总：{(OUT_DIR / 'per_split.csv').resolve()}")
    print(f"9 次汇总：{(OUT_DIR / 'summary_9runs.csv').resolve()}")
    print(f"划分级汇总：{(OUT_DIR / 'summary_split_mean.csv').resolve()}")


if __name__ == "__main__":
    main()
