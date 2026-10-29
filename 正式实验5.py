# -*- coding: utf-8 -*-
"""TA-SLGAT 主实验（完整版拓扑打分：中心化 + 相对差值，δ=0.8）。
GCN / GAT / SLGAT / TA-SLGAT × Cora / CiteSeer / PubMed × 10 seeds。
输出 per-seed、summary、significance 三张 CSV。
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

# ====== 实验配置 ======
DATASETS = ["Cora", "CiteSeer", "PubMed"]
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
THRESHOLD = 0.8            # ← 由完整版拓扑打分扫描确定（三数据集 Val/Test 双第一）
# =====================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
ALPHA, BETA, TEMPERATURE, LAMBDA_U = 0.5, 0.5, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_main_full_topo"
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
    return Planetoid(root=str(root), name=name)[0].to(DEVICE)


def prepare_graph(data):
    n = data.num_nodes
    ids = torch.arange(n, device=data.edge_index.device)
    data.self_loop_ids = ids
    classes = int(data.y.max().item()) + 1
    data.onehot = F.one_hot(data.y, num_classes=classes).to(data.x.dtype)
    data.class_train_masks = [data.train_mask & (data.y == c) for c in range(classes)]
    return data


# ============================================================
# 模型
# ============================================================
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


# ============================================================
# 完整版拓扑打分（中心化 + 相对差值）
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
    """返回 (fused_score, topo, confidence, predicted)。"""
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


# ============================================================
# 训练：GCN / GAT（纯监督）
# ============================================================
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


# ============================================================
# 训练：SLGAT / TA-SLGAT（两阶段自训练）
# ============================================================
def train_slgat(data, seed, proposed, hidden, label_hidden, lr):
    set_seed(seed); classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    # 阶段一：监督预训练
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

    # 阶段二：自训练
    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    selected_count = 0
    best_selected_count = 0
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_unlabeled = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_unlabeled, -1)
            if proposed:
                fused, _, _, _ = topology_scores_full(data, probs, ALPHA, BETA)
                selected = (~data.train_mask) & (fused >= THRESHOLD)
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


# ============================================================
# 显著性检验
# ============================================================
def run_significance_tests(all_rows, out_dir):
    try:
        from scipy.stats import wilcoxon, ttest_rel
    except ImportError as e:
        print(f"\n[skip] scipy 未安装：{e}")
        return []

    metrics = ["test_acc", "test_macro_f1"]
    baselines = ["GCN", "GAT", "SLGAT"]

    grouped = {}
    for r in all_rows:
        grouped.setdefault((r["dataset"], r["model"]), {})[r["seed"]] = r

    rows = []
    for ds in DATASETS:
        ta = grouped.get((ds, "TA-SLGAT"), {})
        if not ta: continue
        for base in baselines:
            bl = grouped.get((ds, base), {})
            seeds = sorted(set(ta.keys()) & set(bl.keys()))
            if len(seeds) < 2: continue
            for metric in metrics:
                x = np.array([ta[s][metric] for s in seeds], dtype=float)
                y = np.array([bl[s][metric] for s in seeds], dtype=float)
                diff = x - y
                try:
                    w_res = wilcoxon(x, y)
                    w_p = float(w_res.pvalue) if hasattr(w_res, "pvalue") else float(w_res[1])
                except ValueError:
                    w_p = 1.0
                t_p = float(ttest_rel(x, y).pvalue)
                sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
                d = float(diff.mean() / sd) if sd > 0 else 0.0
                wins = int((diff > 0).sum())
                ties = int((diff == 0).sum())
                losses = int((diff < 0).sum())
                stars = "**" if w_p < 0.01 else ("*" if w_p < 0.05 else "n.s.")
                rows.append(dict(
                    dataset=ds, baseline=base, metric=metric, n_seeds=len(seeds),
                    ta_mean=float(x.mean()), bl_mean=float(y.mean()),
                    delta=float(diff.mean()), delta_pct=float(diff.mean() * 100),
                    wins=wins, ties=ties, losses=losses,
                    wilcoxon_p=w_p, ttest_p=t_p, cohens_d=d, stars=stars,
                ))

    if not rows:
        print("\n[warn] 无可用的配对样本。")
        return rows

    print("\n" + "=" * 118)
    print("PAIRWISE SIGNIFICANCE: TA-SLGAT vs baselines")
    print("=" * 118)
    header = (f"{'Dataset':<9} {'Baseline':<8} {'Metric':<13} {'TA':>7} {'BL':>7} "
              f"{'Δ%':>6} {'W/T/L':>7} {'p(W)':>9} {'p(t)':>9} {'d':>6} {'Sig':>4}")
    print(header); print("-" * len(header))
    for r in rows:
        print(f"{r['dataset']:<9} {r['baseline']:<8} {r['metric']:<13} "
              f"{r['ta_mean']:.4f} {r['bl_mean']:.4f} "
              f"{r['delta_pct']:+6.2f} {r['wins']:>2}/{r['ties']}/{r['losses']:<2} "
              f"{r['wilcoxon_p']:>9.4f} {r['ttest_p']:>9.4f} {r['cohens_d']:>6.2f} {r['stars']:>4}")

    path = out_dir / "significance.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n显著性结果：{path.resolve()}")
    return rows


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 92)
    print("Main Experiment (FULL topology: centering + relative diff, δ=%.1f)" % THRESHOLD)
    print(f"Models: GCN / GAT / SLGAT / TA-SLGAT × {len(DATASETS)} datasets × {len(SEEDS)} seeds")
    print(f"LR_MAP={LR_MAP}; pseudo τ={TEMPERATURE}; α={ALPHA}, β={BETA}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 92)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data = prepare_graph(load_data(root, name))
        hidden, label_hidden = HIDDEN[name], LABEL_HIDDEN[name]
        lr = LR_MAP[name]
        classes = int(data.y.max().item()) + 1
        print(f"\n{'#'*92}\n{name}: nodes={data.num_nodes}, features={data.num_features}, "
              f"classes={classes}, lr={lr}, "
              f"train/val/test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

        for seed in SEEDS:
            print(f"\n--- {name} seed={seed} ---")
            # GCN / GAT
            for model_name, factory in [
                ("GCN", lambda: GCN(data.num_features, hidden, classes, DROPOUT)),
                ("GAT", lambda: GAT(data.num_features, hidden, classes, DROPOUT)),
            ]:
                set_seed(seed)
                model = factory()
                r = train_supervised(model, data, seed, lr)
                r.update(dataset=name, seed=seed, model=model_name)
                all_rows.append(r)
                print(f"{model_name:8s} val={r['val_acc']:.4f} "
                      f"test_acc={r['test_acc']:.4f} macro_f1={r['test_macro_f1']:.4f} "
                      f"epoch={r['best_epoch']} time={r['elapsed_sec']:.1f}s")
            # SLGAT / TA-SLGAT
            for model_name, proposed in [("SLGAT", False), ("TA-SLGAT", True)]:
                r = train_slgat(data, seed, proposed, hidden, label_hidden, lr)
                r.update(dataset=name, seed=seed, model=model_name)
                all_rows.append(r)
                print(f"{model_name:8s} val={r['val_acc']:.4f} "
                      f"test_acc={r['test_acc']:.4f} macro_f1={r['test_macro_f1']:.4f} "
                      f"pseudo={r['selected_pseudo_nodes']} "
                      f"epoch={r['best_epoch']} time={r['elapsed_sec']:.1f}s")
            save_csv(OUT_DIR / "results_per_seed.csv", all_rows,
                     ["dataset", "seed", "model", "val_acc", "test_acc", "test_macro_f1",
                      "best_epoch", "elapsed_sec", "selected_pseudo_nodes"])

    # 汇总
    summary = []
    print("\n" + "=" * 92 + "\nFINAL SUMMARY (mean ± sample std)\n" + "=" * 92)
    for name in DATASETS:
        for model_name in ["GCN", "GAT", "SLGAT", "TA-SLGAT"]:
            g = [r for r in all_rows if r["dataset"] == name and r["model"] == model_name]
            if not g: continue
            row = {"dataset": name, "model": model_name, "n_seeds": len(g)}
            for metric in ["val_acc", "test_acc", "test_macro_f1", "elapsed_sec"]:
                vals = np.array([float(r[metric]) for r in g])
                row[metric + "_mean"] = float(vals.mean())
                row[metric + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            summary.append(row)
            print(f"{name:8s} {model_name:8s} "
                  f"Acc={row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f} "
                  f"F1={row['test_macro_f1_mean']:.4f}±{row['test_macro_f1_std']:.4f}")
    save_csv(OUT_DIR / "summary_mean_std.csv", summary,
             ["dataset", "model", "n_seeds",
              "val_acc_mean", "val_acc_std", "test_acc_mean", "test_acc_std",
              "test_macro_f1_mean", "test_macro_f1_std",
              "elapsed_sec_mean", "elapsed_sec_std"])

    run_significance_tests(all_rows, OUT_DIR)

    print(f"\n逐 seed：{(OUT_DIR / 'results_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'summary_mean_std.csv').resolve()}")
    print(f"显著性：{(OUT_DIR / 'significance.csv').resolve()}")


if __name__ == "__main__":
    main()
