# -*- coding: utf-8 -*-
"""TA-SLGAT 消融实验：只跑 Confidence only 和 Topology only。
Cora + CiteSeer × 10 seeds，共 40 次训练。
输出 ablation_per_seed.csv / ablation_summary.csv；
若存在主实验 results_per_seed.csv，自动拼成 4 行完整消融表。
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

# ====== 消融配置 ======
DATASETS = ["Cora", "CiteSeer"]          # PubMed 不做消融
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
ABLATION = [
    ("Confidence-only", 1.0, 0.0),       # ② 只用置信度
    ("Topology-only",   0.0, 1.0),       # ③ 只用拓扑
]
# ① SLGAT 与 ④ Full TA (0.5, 0.5) 从主实验读取，不重跑
# =====================

HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
LR_MAP = {"Cora": 0.005, "CiteSeer": 0.005, "PubMed": 0.05}
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
THRESHOLD, TEMPERATURE, LAMBDA_U = 0.9, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_ablation"
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
    ds = Planetoid(root=str(root), name=name)
    data = ds[0].to(DEVICE)
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
    """[ABLATION] alpha/beta 作为参数传入，便于消融切换。"""
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
    """[ABLATION] 与主实验 train_slgat(proposed=True) 唯一区别：alpha/beta 可配。"""
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
            else:
                wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    if best_state is None: raise RuntimeError("预训练没有保存最佳参数")
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


def try_load_main_results():
    """尝试读取主实验 results_per_seed.csv，用于拼 ① SLGAT 和 ④ Full TA。"""
    candidates = [
        Path.cwd() / "ta_slgat_four_model_results" / "results_per_seed.csv",
        Path.cwd() / "data" / "ta_slgat_four_model_results" / "results_per_seed.csv",
    ]
    for p in candidates:
        if p.is_file():
            import pandas as pd
            df = pd.read_csv(p)
            print(f"[ok] 已读取主实验结果：{p.resolve()}")
            return df
    print("[warn] 未找到主实验 results_per_seed.csv，消融表将只含 ② ③ 两行；"
          "①④ 请手动从主实验 summary 复制。")
    return None


def main():
    print("=" * 88)
    print("TA-SLGAT Ablation | Confidence-only vs Topology-only")
    print(f"Datasets={DATASETS}; seeds={len(SEEDS)}; "
          f"α/β: {[(n,a,b) for n,a,b in ABLATION]}; δ={THRESHOLD}")
    print(f"Device={DEVICE}; output={OUT_DIR.resolve()}")
    print("=" * 88)

    root = data_root()
    all_rows = []

    for name in DATASETS:
        data = prepare_graph(load_data(root, name))
        hidden, label_hidden, lr = HIDDEN[name], LABEL_HIDDEN[name], LR_MAP[name]
        classes = int(data.y.max().item()) + 1
        print(f"\n{'#'*88}\n{name}: nodes={data.num_nodes}, features={data.num_features}, "
              f"classes={classes}, lr={lr}")

        for cfg_name, alpha, beta in ABLATION:
            print(f"\n--- {name} | {cfg_name} (α={alpha}, β={beta}) ---")
            for seed in SEEDS:
                r = train_ta_slgat(data, seed, hidden, label_hidden, lr, alpha, beta)
                r.update(dataset=name, config=cfg_name, alpha=alpha, beta=beta, seed=seed)
                all_rows.append(r)
                print(f"  seed={seed:>4d} | val={r['val_acc']:.4f} "
                      f"test={r['test_acc']:.4f} f1={r['test_macro_f1']:.4f} "
                      f"pseudo={r['selected_pseudo_nodes']:>5d} "
                      f"best_ep={r['best_epoch']:>3d}")

        # 每组配置汇总
        for cfg_name, alpha, beta in ABLATION:
            g = [r for r in all_rows if r["dataset"] == name and r["config"] == cfg_name]
            a = np.array([r["test_acc"] for r in g]); f = np.array([r["test_macro_f1"] for r in g])
            print(f"  >>> {name} {cfg_name}: "
                  f"Acc={a.mean()*100:.2f}±{a.std(ddof=1)*100:.2f} "
                  f"F1={f.mean()*100:.2f}±{f.std(ddof=1)*100:.2f}")

    # 逐 seed CSV
    save_csv(OUT_DIR / "ablation_per_seed.csv", all_rows,
             ["dataset", "config", "alpha", "beta", "seed", "val_acc", "test_acc",
              "test_macro_f1", "best_epoch", "selected_pseudo_nodes", "elapsed_sec"])

    # 汇总 CSV（本脚本跑的 ② ③）
    summary = []
    for name in DATASETS:
        for cfg_name, alpha, beta in ABLATION:
            g = [r for r in all_rows if r["dataset"] == name and r["config"] == cfg_name]
            a = np.array([r["test_acc"] for r in g]); f = np.array([r["test_macro_f1"] for r in g])
            summary.append(dict(
                dataset=name, config=cfg_name, alpha=alpha, beta=beta, n_seeds=len(g),
                acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)),
                f1_mean=float(f.mean()), f1_std=float(f.std(ddof=1)),
            ))
    save_csv(OUT_DIR / "ablation_summary.csv", summary,
             ["dataset", "config", "alpha", "beta", "n_seeds",
              "acc_mean", "acc_std", "f1_mean", "f1_std"])

    # 拼完整 4 行消融表
    main_df = try_load_main_results()
    print("\n" + "=" * 88)
    print("FULL ABLATION TABLE (4 rows)")
    print("=" * 88)

    final_rows = []
    for name in DATASETS:
        # ① SLGAT
        if main_df is not None:
            g = main_df[(main_df.dataset == name) & (main_df.model == "SLGAT")]
            if len(g):
                final_rows.append(dict(dataset=name, config="SLGAT (w/o selection)",
                                       alpha="-", beta="-",
                                       acc_mean=float(g["test_acc"].mean())*100,
                                       acc_std=float(g["test_acc"].std(ddof=1))*100))
        # ② ③
        for cfg_name, alpha, beta in ABLATION:
            g = [r for r in all_rows if r["dataset"] == name and r["config"] == cfg_name]
            a = np.array([r["test_acc"] for r in g])
            final_rows.append(dict(dataset=name, config=cfg_name, alpha=alpha, beta=beta,
                                   acc_mean=float(a.mean())*100,
                                   acc_std=float(a.std(ddof=1))*100))
        # ④ Full TA
        if main_df is not None:
            g = main_df[(main_df.dataset == name) & (main_df.model == "TA-SLGAT")]
            if len(g):
                final_rows.append(dict(dataset=name, config="Full TA (0.5/0.5)",
                                       alpha=0.5, beta=0.5,
                                       acc_mean=float(g["test_acc"].mean())*100,
                                       acc_std=float(g["test_acc"].std(ddof=1))*100))

    header = f"{'Dataset':<9} {'Config':<26} {'α':>5} {'β':>5} {'Acc (mean±std)':>20}"
    print(header); print("-" * len(header))
    for r in final_rows:
        print(f"{r['dataset']:<9} {r['config']:<26} {str(r['alpha']):>5} {str(r['beta']):>5} "
              f"{r['acc_mean']:>9.2f}±{r['acc_std']:<5.2f}")

    save_csv(OUT_DIR / "ablation_full_table.csv", final_rows,
             ["dataset", "config", "alpha", "beta", "acc_mean", "acc_std"])
    print(f"\n逐 seed：{(OUT_DIR / 'ablation_per_seed.csv').resolve()}")
    print(f"汇总：   {(OUT_DIR / 'ablation_summary.csv').resolve()}")
    print(f"完整表： {(OUT_DIR / 'ablation_full_table.csv').resolve()}")


if __name__ == "__main__":
    main()
