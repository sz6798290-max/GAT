# -*- coding: utf-8 -*-
"""TA-SLGAT 阈值扫描（完整版 topology_scores：中心化 + 相对差值）。
三数据集 × 5 阈值 × 10 seeds = 150 次训练。
输出 per-seed / summary / best_threshold。
选择依据：val_acc；test_acc 仅报告。
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
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, scatter

# ====== 配置 ======
DATASETS = ["Cora", "CiteSeer", "PubMed"]
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
# ================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
ALPHA, BETA, TEMPERATURE, LAMBDA_U = 0.5, 0.5, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_threshold_sweep_full"
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


def validate_now(epoch, max_epochs):
    return epoch == 0 or (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == max_epochs


def data_root():
    env_root = os.environ.get("GNN_DATA_ROOT")
    root = Path(env_root) if env_root else (Path.cwd() / "data")
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_data(root, name):
    return Planetoid(root=str(root), name=name)[0].to(DEVICE)


def prepare_graph(data):
    """只准备 class_train_masks 和 onehot；邻居分布与原型在 topology 函数内现算。"""
    n = data.num_nodes
    classes = int(data.y.max().item()) + 1
    data.onehot = F.one_hot(data.y, num_classes=classes).to(data.x.dtype)
    data.class_train_masks = [data.train_mask & (data.y == c) for c in range(classes)]
    # 缓存边索引，供 _build_neighbor_dist 使用
    ids = torch.arange(n, device=data.edge_index.device)
    data.self_loop_ids = ids
    return data


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


# ============================================================
# 完整版 topology_scores：中心化 + 相对差值
# ============================================================
def build_neighbor_dist(data, probs, num_classes):
    src, dst = data.edge_index
    label_signal = probs.detach().clone()
    true_onehot = F.one_hot(data.y, num_classes=num_classes).to(dtype=probs.dtype)
    label_signal[data.train_mask] = true_onehot[data.train_mask]
    ids = data.self_loop_ids
    src_all = torch.cat([src, ids])
    dst_all = torch.cat([dst, ids])
    sums = scatter(label_signal[src_all], dst_all, dim=0,
                   dim_size=data.num_nodes, reduce="sum")
    counts = scatter(torch.ones(dst_all.numel(), device=probs.device, dtype=probs.dtype),
                     dst_all, dim=0, dim_size=data.num_nodes, reduce="sum").clamp_min(1.0)
    return sums / counts.unsqueeze(-1)


def topology_scores_full(data, probs, alpha, beta):
    num_classes = data.onehot.size(-1)
    neighbor_dist = build_neighbor_dist(data, probs, num_classes)

    prototypes = torch.zeros(num_classes, num_classes,
                             device=probs.device, dtype=probs.dtype)
    for c in range(num_classes):
        class_mask = data.train_mask & (data.y == c)
        if class_mask.any():
            prototypes[c] = neighbor_dist[class_mask].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(torch.tensor(c, device=probs.device),
                                       num_classes=num_classes).to(dtype=probs.dtype)

    # 中心化
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

    topo = (pred_cos - other_max + 2.0) / 4.0
    topo = topo.clamp(0.0, 1.0)

    return alpha * confidence + beta * topo, topo, confidence, predicted


def train_ta_slgat(data, seed, hidden, label_hidden, lr, threshold):
    set_seed(seed)
    classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    best_val, best_state, wait = -1.0, None, 0
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
                best_val, best_state, wait = val, clone_state(model), 0
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    if best_state is None: raise RuntimeError("预训练未保存参数")
    model.load_state_dict(best_state)
    pretrain_actual_epochs = epoch + 1

    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    best_sel_count = 0
    best_diag = {}
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_unlabeled = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_unlabeled, -1)
            fused, topo, conf, pred = topology_scores_full(data, probs, ALPHA, BETA)
            selected = (~data.train_mask) & (fused >= threshold)
            sel_count = int(selected.sum().item())
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
                best_sel_count = sel_count
                # 诊断：selected ∩ test 的伪标签精度
                test_sel = selected & data.test_mask
                if test_sel.any():
                    pseudo_acc = float((probs[test_sel].argmax(-1) == data.y[test_sel]).float().mean())
                else:
                    pseudo_acc = -1.0
                best_diag = dict(pseudo_acc=pseudo_acc,
                                 test_sel=int(test_sel.sum().item()))
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)

    n_unlabeled = int((~data.train_mask).sum().item())
    out = dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
               best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
               selected_count=best_sel_count,
               selected_ratio=best_sel_count / n_unlabeled)
    out.update(best_diag)
    return out


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    print("=" * 96)
    print("TA-SLGAT Threshold Sweep (FULL topology: centering + relative diff)")
    print(f"Datasets={DATASETS}; thresholds={THRESHOLDS}; seeds={len(SEEDS)}")
    print(f"α={ALPHA}, β={BETA}; pseudo τ={TEMPERATURE}; LR_MAP={LR_MAP}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("提示：阈值选择依据 val_acc；test_acc 仅用于报告。")
    print("=" * 96)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data = prepare_graph(load_data(root, name))
        hidden, label_hidden, lr = HIDDEN[name], LABEL_HIDDEN[name], LR_MAP[name]
        classes = int(data.y.max().item()) + 1
        n_unlabeled = int((~data.train_mask).sum().item())
        print(f"\n{'#'*96}\n{name}: nodes={data.num_nodes}, features={data.num_features}, "
              f"classes={classes}, lr={lr}, unlabeled={n_unlabeled}")

        for thr in THRESHOLDS:
            print(f"\n--- {name} | threshold={thr} ---")
            for seed in SEEDS:
                r = train_ta_slgat(data, seed, hidden, label_hidden, lr, thr)
                r.update(dataset=name, threshold=thr, seed=seed)
                all_rows.append(r)
                print(f"  seed={seed:>4d} | val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
                      f"f1={r['test_macro_f1']:.4f} | sel={r['selected_count']:>5d}"
                      f"({r['selected_ratio']*100:5.1f}%) | pseudo_acc={r.get('pseudo_acc', -1):+.4f} | "
                      f"best_ep={r['best_epoch']:>3d}")
            g = [x for x in all_rows if x["dataset"] == name and x["threshold"] == thr]
            v = np.array([x["val_acc"] for x in g])
            t = np.array([x["test_acc"] for x in g])
            s = np.array([x["selected_ratio"] for x in g])
            p = np.array([x.get("pseudo_acc", -1) for x in g])
            print(f"  >>> thr={thr}: Val={v.mean()*100:.2f}±{v.std(ddof=1)*100:.2f}  "
                  f"Test={t.mean()*100:.2f}±{t.std(ddof=1)*100:.2f}  "
                  f"Sel={s.mean()*100:.1f}%  PseudoAcc={p.mean():+.4f}")

        save_csv(OUT_DIR / "threshold_sweep_per_seed.csv", all_rows,
                 ["dataset", "threshold", "seed", "val_acc", "test_acc", "test_macro_f1",
                  "best_epoch", "selected_count", "selected_ratio", "test_sel",
                  "pseudo_acc", "elapsed_sec"])

    # ====== 汇总 ======
    summary = []
    print("\n" + "=" * 96)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 96)
    header = (f"{'Dataset':<9} {'Thr':>5} | {'Val':>16} | {'Test':>16} | "
              f"{'Sel%':>7} | {'PseudoAcc':>10}")
    print(header); print("-" * len(header))
    for name in DATASETS:
        for thr in THRESHOLDS:
            g = [x for x in all_rows if x["dataset"] == name and x["threshold"] == thr]
            v = np.array([x["val_acc"] for x in g])
            t = np.array([x["test_acc"] for x in g])
            s = np.array([x["selected_ratio"] for x in g])
            p = np.array([x.get("pseudo_acc", -1) for x in g])
            row = dict(dataset=name, threshold=thr, n_seeds=len(g),
                       val_mean=float(v.mean())*100, val_std=float(v.std(ddof=1))*100,
                       test_mean=float(t.mean())*100, test_std=float(t.std(ddof=1))*100,
                       sel_ratio_mean=float(s.mean()),
                       pseudo_acc_mean=float(p.mean()))
            summary.append(row)
            print(f"{name:<9} {thr:>5.1f} | {row['val_mean']:>7.2f}±{row['val_std']:<5.2f} | "
                  f"{row['test_mean']:>7.2f}±{row['test_std']:<5.2f} | "
                  f"{row['sel_ratio_mean']*100:>5.1f}%  | {row['pseudo_acc_mean']:>+9.4f}")

    save_csv(OUT_DIR / "threshold_sweep_summary.csv", summary,
             ["dataset", "threshold", "n_seeds", "val_mean", "val_std",
              "test_mean", "test_std", "sel_ratio_mean", "pseudo_acc_mean"])

    # ====== 每个数据集的验证集最优阈值 ======
    print("\n" + "=" * 96)
    print("BEST THRESHOLD PER DATASET (selected by val_acc)")
    print("=" * 96)
    best_rows = []
    for name in DATASETS:
        cand = [r for r in summary if r["dataset"] == name]
        best = max(cand, key=lambda r: r["val_mean"])
        print(f"{name:<9} → best δ = {best['threshold']}  "
              f"(Val {best['val_mean']:.2f}, Test {best['test_mean']:.2f}, "
              f"Sel {best['sel_ratio_mean']*100:.1f}%)")
        best_rows.append(best)
    save_csv(OUT_DIR / "best_threshold.csv", best_rows,
             ["dataset", "threshold", "val_mean", "val_std",
              "test_mean", "test_std", "sel_ratio_mean", "pseudo_acc_mean"])

    print(f"\n逐 seed：{(OUT_DIR / 'threshold_sweep_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'threshold_sweep_summary.csv').resolve()}")
    print(f"最优：   {(OUT_DIR / 'best_threshold.csv').resolve()}")


if __name__ == "__main__":
    main()
