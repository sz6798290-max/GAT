    # -*- coding: utf-8 -*-
"""GCN / GAT / SLGAT / TA-SLGAT 半监督节点分类正式实验。
只读取本地 Cora、CiteSeer、PubMed 的 PyG processed/data.pt，不自动下载。
输出逐种子结果和 mean±std 汇总 CSV。
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

DATASETS = ["Cora", "CiteSeer", "PubMed"]
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
HIDDEN = {"Cora": 32, "CiteSeer": 32, "PubMed": 16}
LABEL_HIDDEN = {"Cora": 16, "CiteSeer": 16, "PubMed": 8}
DROPOUT, LR, WEIGHT_DECAY = 0.5, 0.005, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
ALPHA, BETA, THRESHOLD, TEMPERATURE, LAMBDA_U = 0.5, 0.5, 0.8, 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path.cwd() / "ta_slgat_four_model_results"
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
    if not mask.any():
        return 0.0
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


def local_root():
    roots = [Path.cwd() / "data", Path(r"D:\dev\vscode机械学习\GNN\data")]
    for root in roots:
        if all((root / d / "processed" / "data.pt").is_file() for d in DATASETS):
            return root
    raise FileNotFoundError(
        "未找到全部本地数据的 processed/data.pt。已停止，避免联网下载。\n" +
        "\n".join(str(r / d / "processed" / "data.pt") for r in roots for d in DATASETS)
    )


def load_data(root, name):
    if not (root / name / "processed" / "data.pt").is_file():
        raise FileNotFoundError(f"本地数据不存在：{root / name / 'processed' / 'data.pt'}")
    # 先确认 processed 缓存存在，再由 PyG 读取；不会触发下载流程。
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


def train_supervised(model, data, seed):
    set_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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


def train_slgat(data, seed, proposed, hidden, label_hidden):
    set_seed(seed); classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, hidden, label_hidden, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    # 阶段一：有标签训练节点监督预训练
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

    # 阶段二：SLGAT 使用所有非训练节点；TA-SLGAT 仅使用过阈值节点
    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    selected_count = 0
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(data.x, data.edge_index)[0], -1)
            if proposed:
                score = topology_scores(data, probs)
                selected = (~data.train_mask) & (score >= THRESHOLD)
            else:
                selected = ~data.train_mask
            selected_count = int(selected.sum().item())
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
                best_val, best_state, best_epoch, wait = val, clone_state(model), PRETRAIN_EPOCHS + epoch + 1, 0
            else: wait += VALIDATE_EVERY
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f"      {'TA-SLGAT' if proposed else 'SLGAT'} self-epoch={epoch+1}, "
                  f"best-val={best_val:.4f}, selected={selected_count}", flush=True)
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        test_logits = model(data.x, data.edge_index)[0]
        test_acc = acc(test_logits, data.y, data.test_mask)
        test_f1 = macro_f1(test_logits, data.y, data.test_mask, classes)
    return dict(val_acc=best_val, test_acc=test_acc, test_macro_f1=test_f1,
                best_epoch=best_epoch, elapsed_sec=time.perf_counter()-start,
                selected_pseudo_nodes=selected_count)


def main():
    print("=" * 88)
    print("GCN / GAT / SLGAT / TA-SLGAT | Cora, CiteSeer, PubMed | 10 seeds")
    print(f"Device={DEVICE}; threads={torch.get_num_threads()}; output={OUT_DIR.resolve()}")
    print(f"TA formula: score={ALPHA}*confidence+{BETA}*topology; threshold={THRESHOLD}")
    root = local_root()
    print(f"本地数据根目录：{root}；不会自动下载数据。")
    all_rows = []

    for name in DATASETS:
        data = prepare_graph(load_data(root, name))
        hidden, label_hidden = HIDDEN[name], LABEL_HIDDEN[name]
        classes = int(data.y.max().item()) + 1
        print(f"\n{'#'*88}\n{name}: nodes={data.num_nodes}, edge_index entries={data.edge_index.size(1)}, "
              f"features={data.num_features}, classes={classes}, "
              f"train/val/test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
        for seed in SEEDS:
            print(f"\n--- {name} seed={seed} ---")
            specs = [
                ("GCN", lambda: GCN(data.num_features, hidden, classes, DROPOUT)),
                ("GAT", lambda: GAT(data.num_features, hidden, classes, DROPOUT)),
            ]
            for model_name, factory in specs:
                # 在初始化模型参数之前设置种子，确保每次运行可复现。
                set_seed(seed)
                model = factory()
                result = train_supervised(model, data, seed)
                row = {"dataset": name, "seed": seed, "model": model_name, **result}
                all_rows.append(row)
                print(f"{model_name:8s} val={result['val_acc']:.4f} "
                      f"test_acc={result['test_acc']:.4f} macro_f1={result['test_macro_f1']:.4f} "
                      f"epoch={result['best_epoch']} time={result['elapsed_sec']:.1f}s")
            for model_name, proposed in [("SLGAT", False), ("TA-SLGAT", True)]:
                result = train_slgat(data, seed, proposed, hidden, label_hidden)
                row = {"dataset": name, "seed": seed, "model": model_name, **result}
                all_rows.append(row)
                print(f"{model_name:8s} val={result['val_acc']:.4f} "
                      f"test_acc={result['test_acc']:.4f} macro_f1={result['test_macro_f1']:.4f} "
                      f"pseudo_nodes={result['selected_pseudo_nodes']} "
                      f"best_epoch={result['best_epoch']} time={result['elapsed_sec']:.1f}s")

            # 每完成一个 seed 就保存一次，防止长实验中断丢失结果
            save_per_seed(all_rows)

    summary = []
    print("\n" + "=" * 88 + "\nFINAL SUMMARY: mean ± sample std\n" + "=" * 88)
    for name in DATASETS:
        for model_name in ["GCN", "GAT", "SLGAT", "TA-SLGAT"]:
            group = [r for r in all_rows if r["dataset"] == name and r["model"] == model_name]
            if not group: continue
            row = {"dataset": name, "model": model_name, "n_seeds": len(group)}
            for metric in ["val_acc", "test_acc", "test_macro_f1", "elapsed_sec"]:
                values = np.asarray([float(r[metric]) for r in group])
                row[metric + "_mean"] = float(values.mean())
                row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            summary.append(row)
            print(f"{name:8s} {model_name:8s} "
                  f"Acc={row['test_acc_mean']:.4f}±{row['test_acc_std']:.4f} "
                  f"Macro-F1={row['test_macro_f1_mean']:.4f}±{row['test_macro_f1_std']:.4f}")
    with (OUT_DIR / "summary_mean_std.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader(); writer.writerows(summary)
    print(f"\n逐种子结果：{(OUT_DIR / 'results_per_seed.csv').resolve()}")
    print(f"汇总结果：{(OUT_DIR / 'summary_mean_std.csv').resolve()}")


def save_per_seed(rows):
    path = OUT_DIR / "results_per_seed.csv"
    fields = ["dataset", "seed", "model", "val_acc", "test_acc", "test_macro_f1",
              "best_epoch", "elapsed_sec", "selected_pseudo_nodes"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
