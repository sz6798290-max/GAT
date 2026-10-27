# -*- coding: utf-8 -*-
"""TA-SLGAT 阈值扫描实验（单数据集）。
只跑 TA-SLGAT，扫描 THRESHOLDS 列表，固定同一组 SEEDS，输出:
  - threshold_sweep_per_seed.csv   每个 (threshold, seed) 一行
  - threshold_sweep_summary.csv    每个 threshold 汇总一行
同时记录 best epoch 的筛选比例 (sel%) 与伪标签在 test 节点上的精度 (pseudo_acc)。
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

# ============ 扫描配置（只改这里） ============
DATASET = "Cora"                       # 单数据集，先 Cora
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [34, 72, 198, 264, 588, 699, 795, 854, 961, 1024]
# ==============================================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.05, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
ALPHA, BETA, TEMPERATURE, LAMBDA_U = 0.5, 0.5, 0.1, 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_threshold_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------- 通用工具 ----------------
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
    vals = []
    for c in range(classes):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        den = 2 * tp + fp + fn
        vals.append(2 * tp / den if den else 0.0)
    return float(np.mean(vals))


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


# ---------------- 模型（与主实验保持一致） ----------------
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


# ---------------- 单次 TA-SLGAT 训练（带阈值 + 诊断） ----------------
def train_ta_slgat_threshold(data, seed, hidden, label_hidden, threshold, lr):
    set_seed(seed)
    classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    # 阶段一：监督预训练
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
            else:
                wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    if best_state is None: raise RuntimeError("预训练没有保存最佳参数")
    model.load_state_dict(best_state)
    pretrain_actual_epochs = epoch + 1

    # 阶段二：TA-SLGAT 自训练（阈值 = threshold）
    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    best_sel_count, best_pseudo_acc, best_test_sel = 0, -1.0, 0
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(data.x, data.edge_index)[0], -1)
            score = topology_scores(data, probs)
            selected = (~data.train_mask) & (score >= threshold)
            sel_count = int(selected.sum().item())
            targets = F.softmax(torch.log(probs.clamp_min(1e-12)) / TEMPERATURE, -1)
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
                # 诊断：best epoch 时 selected ∩ test 的伪标签精度
                test_sel = selected & data.test_mask
                if test_sel.any():
                    best_pseudo_acc = float(
                        (probs[test_sel].argmax(-1) == data.y[test_sel]).float().mean()
                    )
                    best_test_sel = int(test_sel.sum().item())
                else:
                    best_pseudo_acc, best_test_sel = -1.0, 0
            else:
                wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)

    n_unlabeled = int((~data.train_mask).sum().item())
    return dict(
        val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
        best_epoch=best_epoch,
        selected_count=best_sel_count,
        selected_ratio=best_sel_count / n_unlabeled,
        test_selected=best_test_sel,
        pseudo_acc=best_pseudo_acc,
        elapsed_sec=time.perf_counter() - start,
    )


# ---------------- 主流程 ----------------
def main():
    print("=" * 88)
    print(f"TA-SLGAT Threshold Sweep | dataset={DATASET} | thresholds={THRESHOLDS}")
    print(f"Device={DEVICE}; seeds={len(SEEDS)}; output={OUT_DIR.resolve()}")
    print("=" * 88)

    root = data_root()
    data = prepare_graph(load_data(root, DATASET))
    hidden, label_hidden, lr = HIDDEN[DATASET], LABEL_HIDDEN[DATASET], LR_MAP[DATASET]
    classes = int(data.y.max().item()) + 1
    n_unlabeled = int((~data.train_mask).sum().item())
    print(f"{DATASET}: nodes={data.num_nodes}, features={data.num_features}, classes={classes}, "
          f"train/val/test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}, "
          f"unlabeled={n_unlabeled}, lr={lr}")

    all_rows = []
    for thr in THRESHOLDS:
        print(f"\n{'#'*88}\nThreshold = {thr}\n{'#'*88}")
        thr_rows = []
        for seed in SEEDS:
            r = train_ta_slgat_threshold(data, seed, hidden, label_hidden, thr, lr)
            r["seed"] = seed; r["threshold"] = thr
            thr_rows.append(r); all_rows.append(r)
            print(f"  seed={seed:>4d} | acc={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f} | "
                  f"sel={r['selected_count']:>5d}({r['selected_ratio']*100:5.1f}%) | "
                  f"pseudo_acc={r['pseudo_acc']:+.4f} | best_epoch={r['best_epoch']:>3d}")
        a = np.array([x["test_acc"] for x in thr_rows])
        f = np.array([x["test_macro_f1"] for x in thr_rows])
        s = np.array([x["selected_ratio"] for x in thr_rows])
        p = np.array([x["pseudo_acc"] for x in thr_rows])
        e = np.array([x["best_epoch"] for x in thr_rows])
        print(f"  >>> thr={thr}: Acc={a.mean():.4f}±{a.std(ddof=1):.4f} | "
              f"F1={f.mean():.4f}±{f.std(ddof=1):.4f} | "
              f"Sel={s.mean()*100:.1f}% | PseudoAcc={p.mean():+.4f} | "
              f"BestEpoch(med)={int(np.median(e))}")

    # 最终对比表
    print("\n" + "=" * 88)
    print("FINAL THRESHOLD SWEEP")
    print("=" * 88)
    header = f"{'Thr':>5} | {'Acc (mean±std)':>18} | {'F1 (mean±std)':>18} | {'Sel%':>7} | {'PseudoAcc':>10} | {'BestEpoch(med)':>14}"
    print(header); print("-" * len(header))
    summary_rows = []
    for thr in THRESHOLDS:
        group = [r for r in all_rows if r["threshold"] == thr]
        a = np.array([x["test_acc"] for x in group])
        f = np.array([x["test_macro_f1"] for x in group])
        s = np.array([x["selected_ratio"] for x in group])
        p = np.array([x["pseudo_acc"] for x in group])
        e = np.array([x["best_epoch"] for x in group])
        print(f"{thr:>5.2f} | "
              f"{a.mean():.4f}±{a.std(ddof=1):.4f}   | "
              f"{f.mean():.4f}±{f.std(ddof=1):.4f}   | "
              f"{s.mean()*100:>5.1f}%  | "
              f"{p.mean():>+9.4f}  | "
              f"{int(np.median(e)):>14d}")
        summary_rows.append(dict(
            dataset=DATASET, threshold=thr, n_seeds=len(group),
            acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
            f1_mean=float(f.mean()), f1_std=float(f.std(ddof=1)),
            selected_ratio_mean=float(s.mean()),
            pseudo_acc_mean=float(p.mean()),
            best_epoch_median=int(np.median(e)),
        ))

    # 写 CSV
    with (OUT_DIR / "threshold_sweep_per_seed.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        fields = ["dataset", "threshold", "seed", "val_acc", "test_acc", "test_macro_f1",
                  "best_epoch", "selected_count", "selected_ratio", "test_selected",
                  "pseudo_acc", "elapsed_sec"]
        w = csv.DictWriter(fp, fieldnames=fields); w.writeheader()
        for r in all_rows:
            row = {k: r.get(k) for k in fields}; row["dataset"] = DATASET
            w.writerow(row)
    with (OUT_DIR / "threshold_sweep_summary.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print(f"\n逐 seed 结果：{(OUT_DIR / 'threshold_sweep_per_seed.csv').resolve()}")
    print(f"汇总结果：   {(OUT_DIR / 'threshold_sweep_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
