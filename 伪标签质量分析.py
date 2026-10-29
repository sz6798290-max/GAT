# -*- coding: utf-8 -*-
"""伪标签质量分析：全选（SLGAT） vs 完整 TA（筛选后）。
在自训练 best epoch，统计 test 节点上的伪标签精度与覆盖率。
三数据集 × 2 配置 × 10 seed = 60 次训练。
"""
import os
THREADS = int(os.environ.get("GNN_CPU_THREADS", str(min(2, os.cpu_count() or 1))))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import csv, random
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
THRESHOLD = 0.8
CONFIGS = [
    ("All (SLGAT)", None, None),   # 全选
    ("Full TA",     0.5,  0.5),    # 筛选
]
# ==================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
TEMPERATURE, LAMBDA_U = 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_pseudo_quality"
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


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def validate_now(epoch, max_epochs):
    return epoch == 0 or (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == max_epochs


def data_root():
    env_root = os.environ.get("GNN_DATA_ROOT")
    root = Path(env_root) if env_root else (Path.cwd() / "data")
    root.mkdir(parents=True, exist_ok=True)
    return root


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


def build_neighbor_dist(data, probs, num_classes):
    src, dst = data.edge_index
    label_signal = probs.detach().clone()
    true_onehot = F.one_hot(data.y, num_classes=num_classes).to(dtype=probs.dtype)
    label_signal[data.train_mask] = true_onehot[data.train_mask]
    ids = data.self_loop_ids
    src_all = torch.cat([src, ids]); dst_all = torch.cat([dst, ids])
    sums = scatter(label_signal[src_all], dst_all, dim=0,
                   dim_size=data.num_nodes, reduce="sum")
    counts = scatter(torch.ones(dst_all.numel(), device=probs.device, dtype=probs.dtype),
                     dst_all, dim=0, dim_size=data.num_nodes, reduce="sum").clamp_min(1.0)
    return sums / counts.unsqueeze(-1)


def compute_fused(data, probs, alpha, beta):
    num_classes = data.onehot.size(-1)
    neighbor_dist = build_neighbor_dist(data, probs, num_classes)
    prototypes = torch.zeros(num_classes, num_classes, device=probs.device, dtype=probs.dtype)
    for c in range(num_classes):
        m = data.train_mask & (data.y == c)
        if m.any():
            prototypes[c] = neighbor_dist[m].mean(dim=0)
        else:
            prototypes[c] = F.one_hot(torch.tensor(c, device=probs.device),
                                       num_classes=num_classes).to(dtype=probs.dtype)
    p_bar = prototypes.mean(dim=0, keepdim=True)
    proto_tilde = prototypes - p_bar
    node_norm = F.normalize(neighbor_dist, p=2, dim=-1)
    proto_norm = F.normalize(proto_tilde, p=2, dim=-1)
    cos_all = node_norm @ proto_norm.T
    confidence, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)
    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values
    topo = ((pred_cos - other_max + 2.0) / 4.0).clamp(0.0, 1.0)
    return alpha * confidence + beta * topo


def prepare(data):
    n = data.num_nodes
    data.self_loop_ids = torch.arange(n, device=data.edge_index.device)
    classes = int(data.y.max().item()) + 1
    data.onehot = F.one_hot(data.y, num_classes=classes).to(data.x.dtype)
    return data


def run_once(data, seed, hidden, label_hidden, lr, alpha, beta):
    proposed = alpha is not None
    set_seed(seed)
    classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # 阶段一
    best_val, best_state, wait = -1.0, None, 0
    for epoch in range(PRETRAIN_EPOCHS):
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)[0]
        F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]).backward(); opt.step()
        if validate_now(epoch, PRETRAIN_EPOCHS):
            model.eval()
            with torch.inference_mode():
                val = acc(model(data.x, data.edge_index)[0], data.y, data.val_mask)
            if val > best_val: best_val, best_state, wait = val, clone_state(model), 0
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    model.load_state_dict(best_state)

    # 阶段二
    n_unlabeled = int((~data.train_mask).sum().item())
    test_mask = data.test_mask
    best_val, best_state, wait = -1.0, clone_state(model), 0
    best_diag = None
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_u = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_u, -1)
            if proposed:
                fused = compute_fused(data, probs, alpha, beta)
                selected = (~data.train_mask) & (fused >= THRESHOLD)
            else:
                selected = ~data.train_mask
            pred = probs.argmax(-1)
            targets = F.softmax(logits_u / TEMPERATURE, -1)
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
                # 诊断：selected ∩ test
                sel_test = selected & test_mask
                cov = int(selected.sum().item()) / n_unlabeled
                if sel_test.any():
                    prec = float((pred[sel_test] == data.y[sel_test]).float().mean().item())
                    n_sel_test = int(sel_test.sum().item())
                else:
                    prec, n_sel_test = 0.0, 0
                best_diag = dict(precision=prec, coverage=cov,
                                 sel_count=int(selected.sum().item()),
                                 n_sel_test=n_sel_test)
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        logits = model(data.x, data.edge_index)[0]
        test_acc = acc(logits, data.y, test_mask)

    d = dict(val_acc=best_val, test_acc=test_acc)
    if best_diag:
        d.update(best_diag)
        p, c = best_diag["precision"], best_diag["coverage"]
        d["pseudo_f1"] = 2 * p * c / (p + c) if (p + c) > 0 else 0.0
    else:
        d.update(dict(precision=0.0, coverage=0.0, sel_count=0,
                      n_sel_test=0, pseudo_f1=0.0))
    return d


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    print("=" * 96)
    print("Pseudo-Label Quality: All vs Full TA | δ=%.1f" % THRESHOLD)
    print(f"Datasets={DATASETS}; seeds={len(SEEDS)}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 96)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data = prepare(Planetoid(root=str(root), name=name)[0].to(DEVICE))
        hidden, label_hidden, lr = HIDDEN[name], LABEL_HIDDEN[name], LR_MAP[name]
        n_unlabeled = int((~data.train_mask).sum().item())
        print(f"\n{'#'*96}\n{name}: unlabeled={n_unlabeled}, test={int(data.test_mask.sum())}")

        for cfg_name, alpha, beta in CONFIGS:
            print(f"\n--- {name} | {cfg_name} ---")
            for seed in SEEDS:
                r = run_once(data, seed, hidden, label_hidden, lr, alpha, beta)
                r.update(dataset=name, config=cfg_name, seed=seed)
                all_rows.append(r)
                print(f"  seed={seed:>4d} | 精度={r['precision']:.4f} "
                      f"覆盖率={r['coverage']:.4f} PseudoF1={r['pseudo_f1']:.4f} | "
                      f"筛出={r['sel_count']:>5d} (test中{r['n_sel_test']}个) | "
                      f"test_acc={r['test_acc']:.4f}")

        save_csv(OUT_DIR / "pseudo_quality_per_seed.csv", all_rows,
                 ["dataset", "config", "seed", "precision", "coverage", "pseudo_f1",
                  "sel_count", "n_sel_test", "test_acc", "val_acc"])

    # 汇总
    summary = []
    print("\n" + "=" * 96)
    print("PSEUDO-LABEL QUALITY SUMMARY (10 seeds 均值)")
    print("=" * 96)
    header = (f"{'Dataset':<9} {'Config':<12} | {'Precision':>10} | "
              f"{'Coverage':>9} | {'Pseudo-F1':>10} | {'Test Acc':>9}")
    print(header); print("-" * len(header))
    for name in DATASETS:
        for cfg_name, _, _ in CONFIGS:
            g = [r for r in all_rows if r["dataset"] == name and r["config"] == cfg_name]
            p = np.mean([r["precision"] for r in g])
            c = np.mean([r["coverage"] for r in g])
            pf = np.mean([r["pseudo_f1"] for r in g])
            ta = np.mean([r["test_acc"] for r in g])
            summary.append(dict(dataset=name, config=cfg_name,
                                precision=p, coverage=c, pseudo_f1=pf, test_acc=ta))
            print(f"{name:<9} {cfg_name:<12} | {p:>9.4f} | {c:>8.4f} | "
                  f"{pf:>9.4f} | {ta:>8.4f}")

    save_csv(OUT_DIR / "pseudo_quality_summary.csv", summary,
             ["dataset", "config", "precision", "coverage", "pseudo_f1", "test_acc"])
    print(f"\n逐 seed：{(OUT_DIR / 'pseudo_quality_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'pseudo_quality_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
