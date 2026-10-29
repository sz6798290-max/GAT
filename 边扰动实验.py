# -*- coding: utf-8 -*-
"""边扰动实验：Cora 上随机删除 15% 的边，对比 Confidence-only vs Full TA。
目的：验证在图结构被扰动、置信度信号不可靠时，拓扑项是否提供独立增益。
固定一次扰动（seed=0），10 个训练种子。
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
from torch_geometric.utils import softmax, scatter, to_undirected

# ====== 边扰动实验配置 ======
DATASET = "Cora"
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
DROP_RATIO = 0.15                     # 删除 15% 的边
PERTURB_SEED = 0                      # 固定扰动，确保所有训练种子看到同一个扰动图
CONFIGS = [
    ("Confidence-only", 1.0, 0.0),
    ("Full TA",         0.5, 0.5),
]
# =============================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
THRESHOLD, TEMPERATURE, LAMBDA_U = 0.9, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_edge_perturb"
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


def perturb_edges(data, drop_ratio=0.15, seed=0):
    """随机删除 drop_ratio 比例的无向边；保持对称性。
    做法：取 u < v 的唯一方向，随机保留，再对称化重建。
    """
    ei = to_undirected(data.edge_index)          # 确保无向
    src, dst = ei[0], ei[1]
    mask = src < dst                             # 唯一方向
    u, v = src[mask], dst[mask]
    n_unique = u.numel()
    g = torch.Generator(device=ei.device).manual_seed(seed)
    keep = torch.rand(n_unique, generator=g, device=ei.device) > drop_ratio
    new_u = torch.cat([u[keep], v[keep]])
    new_v = torch.cat([v[keep], u[keep]])
    data.edge_index = torch.stack([new_u, new_v], dim=0)
    print(f"[perturb] 原始无向边 {n_unique} → 保留 {int(keep.sum())} "
          f"（删除 {drop_ratio*100:.0f}%）→ edge_index {data.edge_index.size(1)} 条（双向计）")
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


def topology_scores(data, probs, alpha, beta):
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
    return alpha * confidence + beta * topo


def train_ta_slgat(data, seed, hidden, label_hidden, lr, alpha, beta):
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
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_unlabeled = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_unlabeled, -1)
            score = topology_scores(data, probs, alpha, beta)
            selected = (~data.train_mask) & (score >= THRESHOLD)
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
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)
    return dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
                best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
                selected_pseudo_nodes=best_sel_count)


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    print("=" * 92)
    print(f"Edge Perturbation Experiment | {DATASET}, drop {DROP_RATIO*100:.0f}% edges")
    print(f"seeds={len(SEEDS)}; configs={[(n,a,b) for n,a,b in CONFIGS]}; δ={THRESHOLD}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 92)

    root = data_root()
    data = load_data(root, DATASET)
    n_orig_edges = data.edge_index.size(1)

    # 先扰动边，再 prepare_graph（重算 topology_counts）
    data = perturb_edges(data, drop_ratio=DROP_RATIO, seed=PERTURB_SEED)
    n_pert_edges = data.edge_index.size(1)
    data = prepare_graph(data)

    hidden, label_hidden, lr = HIDDEN[DATASET], LABEL_HIDDEN[DATASET], LR_MAP[DATASET]
    classes = int(data.y.max().item()) + 1
    print(f"{DATASET}: nodes={data.num_nodes}, features={data.num_features}, "
          f"classes={classes}, lr={lr}")
    print(f"edges: {n_orig_edges} → {n_pert_edges} "
          f"(删除了 {(1 - n_pert_edges/n_orig_edges)*100:.1f}% 的 edge_index 条目)")

    all_rows = []
    for cfg_name, alpha, beta in CONFIGS:
        print(f"\n--- {DATASET} | {cfg_name} (α={alpha}, β={beta}) ---")
        for seed in SEEDS:
            r = train_ta_slgat(data, seed, hidden, label_hidden, lr, alpha, beta)
            r.update(dataset=DATASET, config=cfg_name, alpha=alpha, beta=beta, seed=seed)
            all_rows.append(r)
            print(f"  seed={seed:>4d} | val={r['val_acc']:.4f} "
                  f"test={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f} "
                  f"pseudo={r['selected_pseudo_nodes']:>5d} best_ep={r['best_epoch']:>3d}")
        g = [x for x in all_rows if x["config"] == cfg_name]
        a_ = np.array([x["test_acc"] for x in g])
        f_ = np.array([x["test_macro_f1"] for x in g])
        print(f"  >>> {cfg_name}: Acc={a_.mean()*100:.2f}±{a_.std(ddof=1)*100:.2f}  "
              f"F1={f_.mean()*100:.2f}±{f_.std(ddof=1)*100:.2f}")

    # 汇总对比
    print("\n" + "=" * 92)
    print("EDGE PERTURBATION SUMMARY")
    print("=" * 92)
    summary = []
    for cfg_name, alpha, beta in CONFIGS:
        g = [x for x in all_rows if x["config"] == cfg_name]
        a_ = np.array([x["test_acc"] for x in g])
        f_ = np.array([x["test_macro_f1"] for x in g])
        summary.append(dict(
            dataset=DATASET, config=cfg_name, alpha=alpha, beta=beta, n_seeds=len(g),
            acc_mean=float(a_.mean())*100, acc_std=float(a_.std(ddof=1))*100,
            f1_mean=float(f_.mean())*100, f1_std=float(f_.std(ddof=1))*100,
        ))
    header = f"{'Dataset':<8} {'Config':<18} {'α':>4} {'β':>4} | {'Acc':>16} | {'F1':>16}"
    print(header); print("-" * len(header))
    for r in summary:
        print(f"{r['dataset']:<8} {r['config']:<18} {r['alpha']:>4.1f} {r['beta']:>4.1f} | "
              f"{r['acc_mean']:>7.2f}±{r['acc_std']:<5.2f} | "
              f"{r['f1_mean']:>7.2f}±{r['f1_std']:<5.2f}")

    # 计算 Δ
    if len(summary) == 2:
        delta = summary[1]["acc_mean"] - summary[0]["acc_mean"]
        print(f"\nΔ (Full TA − Confidence-only) = {delta:+.2f}% (Acc)")
        print("参考（干净图消融）：Full TA − Confidence-only = "
              f"{84.13 - 84.18:+.2f}% (Cora)")
        if delta >= 0.5:
            print("→ Full TA 在扰动图上有明显优势，拓扑项提供独立增益。")
        elif delta <= -0.5:
            print("→ Full TA 在扰动图上劣于 Confidence-only，拓扑项对噪声边敏感。")
        else:
            print("→ 两组差异在 0.5% 以内，拓扑项在扰动下仍未提供显著独立增益。")

    save_csv(OUT_DIR / "edge_perturb_per_seed.csv", all_rows,
             ["dataset", "config", "alpha", "beta", "seed", "val_acc", "test_acc",
              "test_macro_f1", "best_epoch", "selected_pseudo_nodes", "elapsed_sec"])
    save_csv(OUT_DIR / "edge_perturb_summary.csv", summary,
             ["dataset", "config", "alpha", "beta", "n_seeds",
              "acc_mean", "acc_std", "f1_mean", "f1_std"])

    print(f"\n逐 seed：{(OUT_DIR / 'edge_perturb_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'edge_perturb_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
