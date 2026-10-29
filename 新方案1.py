# -*- coding: utf-8 -*-
"""TA-SLGAT 门控融合实验（方案 B）。
对比：
  (1) Equal-soft:  α=β=0.5 固定，软权重
  (2) Gated-soft:  门控网络逐节点学习 g，score = g*conf + (1-g)*topo，软权重
两组唯一变量是 "α 是否可学"；软权重使门控可微。
输出逐 seed、汇总、门控学到的 g 的统计。
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
DATASETS = ["Cora", "CiteSeer"]                  # 先这两个，PubMed 后补
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
CONFIGS = ["Equal-soft", "Gated-soft"]
# ==================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
THRESHOLD, TEMPERATURE, LAMBDA_U = 0.9, 0.1, 1.0
SOFT_TEMP = 0.1                                   # 软权重温度，越小越接近硬阈值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_gated_fusion"
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


class AdaptiveFusion(nn.Module):
    """门控：输入 (confidence, topo)，输出逐节点 g ∈ (0,1)。
    小初始化使初始 g≈0.5（等权起步）。"""
    def __init__(self, hidden=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
    def forward(self, confidence, topo):
        x = torch.stack([confidence, topo], dim=-1)    # [N, 2]
        return self.gate(x).squeeze(-1)                # [N]


def compute_conf_topo(data, probs):
    """返回 confidence [N] 和 topo [N]。"""
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
    return confidence, topo


def train_config(data, seed, hidden, label_hidden, lr, config):
    """config ∈ {'Equal-soft', 'Gated-soft'}。"""
    use_gate = (config == "Gated-soft")
    set_seed(seed)
    classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    fusion = AdaptiveFusion().to(DEVICE) if use_gate else None
    params = list(model.parameters()) + (list(fusion.parameters()) if fusion else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    # 阶段一：预训练（门控不参与，无梯度）
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

    # 阶段二：自训练（门控参与，软权重）
    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    best_sel_count = 0
    best_g_stats = {}
    non_train_mask = ~data.train_mask
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_unlabeled = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_unlabeled, -1)
            confidence, topo = compute_conf_topo(data, probs)
            targets = F.softmax(logits_unlabeled / TEMPERATURE, -1)

        # 融合（门控可微；confidence/topo 是常数，梯度只流向 fusion）
        if use_gate:
            g = fusion(confidence, topo)                 # [N] ∈ (0,1)
            score = g * confidence + (1.0 - g) * topo
        else:
            g = torch.full_like(confidence, 0.5)
            score = 0.5 * confidence + 0.5 * topo
        # 软权重：score 远高于阈值 → w≈1；远低于 → w≈0
        w = torch.sigmoid((score - THRESHOLD) / SOFT_TEMP)  # [N] ∈ (0,1)
        w = w * non_train_mask.float()                      # 只在非训练节点上
        # 硬筛选仅用于诊断记录
        selected_hard = non_train_mask & (score.detach() >= THRESHOLD)
        sel_count = int(selected_hard.sum().item())

        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)[0]
        loss_sup = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        per_node = -(targets * F.log_softmax(logits, -1)).sum(-1)  # [N]
        loss_unsup = (w * per_node).sum() / w.sum().clamp_min(1e-8)
        (loss_sup + LAMBDA_U * loss_unsup).backward(); opt.step()

        if validate_now(epoch, SELF_TRAIN_EPOCHS):
            model.eval()
            with torch.inference_mode():
                val = acc(model(data.x, data.edge_index)[0], data.y, data.val_mask)
            if val > best_val:
                best_val, best_state, wait = val, clone_state(model), 0
                best_epoch = pretrain_actual_epochs + epoch + 1
                best_sel_count = sel_count
                if use_gate:
                    gg = g[non_train_mask].detach()
                    best_g_stats = dict(
                        g_mean=float(gg.mean()), g_std=float(gg.std()),
                        g_min=float(gg.min()),  g_max=float(gg.max()),
                    )
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)

    out = dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
               best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
               selected_pseudo_nodes=best_sel_count)
    out.update(best_g_stats)
    return out


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    print("=" * 96)
    print("Gated Fusion Experiment (方案 B) | Equal-soft vs Gated-soft")
    print(f"Datasets={DATASETS}; seeds={len(SEEDS)}; δ={THRESHOLD}; soft_temp={SOFT_TEMP}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 96)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data = prepare_graph(load_data(root, name))
        hidden, label_hidden, lr = HIDDEN[name], LABEL_HIDDEN[name], LR_MAP[name]
        classes = int(data.y.max().item()) + 1
        print(f"\n{'#'*96}\n{name}: nodes={data.num_nodes}, features={data.num_features}, "
              f"classes={classes}, lr={lr}")

        for config in CONFIGS:
            print(f"\n--- {name} | {config} ---")
            for seed in SEEDS:
                r = train_config(data, seed, hidden, label_hidden, lr, config)
                r.update(dataset=name, config=config, seed=seed)
                all_rows.append(r)
                extra = ""
                if "g_mean" in r:
                    extra = (f" | g_mean={r['g_mean']:.3f} "
                             f"g_std={r['g_std']:.3f} [{r['g_min']:.3f},{r['g_max']:.3f}]")
                print(f"  seed={seed:>4d} | val={r['val_acc']:.4f} "
                      f"test={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f}"
                      f"{extra}")
            g = [x for x in all_rows if x["dataset"] == name and x["config"] == config]
            v = np.array([x["val_acc"] for x in g])
            t = np.array([x["test_acc"] for x in g])
            print(f"  >>> {name} {config}: "
                  f"Val={v.mean()*100:.2f}±{v.std(ddof=1)*100:.2f}  "
                  f"Test={t.mean()*100:.2f}±{t.std(ddof=1)*100:.2f}")
            if config == "Gated-soft":
                gm = [x.get("g_mean", np.nan) for x in g if "g_mean" in x]
                if gm:
                    print(f"  >>> 门控 g 的均值 = {np.mean(gm):.4f} "
                          f"(0.5=等权，>0.5 偏置信度，<0.5 偏拓扑)")

        save_csv(OUT_DIR / "gated_per_seed.csv", all_rows,
                 ["dataset", "config", "seed", "val_acc", "test_acc", "test_macro_f1",
                  "best_epoch", "selected_pseudo_nodes", "elapsed_sec",
                  "g_mean", "g_std", "g_min", "g_max"])

    # 汇总
    summary = []
    print("\n" + "=" * 96)
    print("SUMMARY: Equal-soft vs Gated-soft")
    print("=" * 96)
    header = f"{'Dataset':<9} {'Config':<12} | {'Val':>16} | {'Test':>16} | {'g_mean':>8}"
    print(header); print("-" * len(header))
    for name in DATASETS:
        for config in CONFIGS:
            g = [x for x in all_rows if x["dataset"] == name and x["config"] == config]
            v = np.array([x["val_acc"] for x in g])
            t = np.array([x["test_acc"] for x in g])
            gm = [x.get("g_mean") for x in g if x.get("g_mean") is not None]
            row = dict(dataset=name, config=config, n_seeds=len(g),
                       val_mean=float(v.mean())*100, val_std=float(v.std(ddof=1))*100,
                       test_mean=float(t.mean())*100, test_std=float(t.std(ddof=1))*100,
                       g_mean=(float(np.mean(gm)) if gm else None))
            summary.append(row)
            gm_str = f"{row['g_mean']:.4f}" if row["g_mean"] is not None else "-"
            print(f"{name:<9} {config:<12} | "
                  f"{row['val_mean']:>7.2f}±{row['val_std']:<5.2f} | "
                  f"{row['test_mean']:>7.2f}±{row['test_std']:<5.2f} | {gm_str:>8}")

    save_csv(OUT_DIR / "gated_summary.csv", summary,
             ["dataset", "config", "n_seeds", "val_mean", "val_std",
              "test_mean", "test_std", "g_mean"])

    print(f"\n逐 seed：{(OUT_DIR / 'gated_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'gated_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
