# -*- coding: utf-8 -*-
"""最小代价验证：Cora 上 Topology-only 的 3 种映射方式对比。
目的：判断放宽 topo 映射后，Topology-only 能否筛出节点、能否超过 SLGAT（82.55）。
不改任何其他配置（δ=0.8, LR_MAP, 温度路径, 验证频率均与主实验一致）。
"""
import os
THREADS = int(os.environ.get("GNN_CPU_THREADS", str(min(2, os.cpu_count() or 1))))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax, scatter

# ====== 配置 ======
DATASET = "Cora"
SEEDS = [34, 72, 198, 284, 588, 699, 795, 854, 961, 1024]
THRESHOLD = 0.8
TOPO_MODES = ["original", "sigmoid_t0.10", "sigmoid_t0.05", "linear_x2.5"]  # 4 种映射
# 参照：SLGAT 在 Cora 上 82.55±0.64
SLGAT_REF = 82.55
# ==================

HIDDEN, LABEL_HIDDEN, LR = 32, 16, 0.005
DROPOUT, WEIGHT_DECAY = 0.5, 5e-4
PRETRAIN_EPOCHS, SELF_TRAIN_EPOCHS, PATIENCE, VALIDATE_EVERY = 200, 600, 100, 5
TEMPERATURE, LAMBDA_U = 0.1, 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    root = env_root if env_root else os.path.join(os.getcwd(), "data")
    os.makedirs(root, exist_ok=True)
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


def prepare(data):
    n = data.num_nodes
    data.self_loop_ids = torch.arange(n, device=data.edge_index.device)
    return data


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


def compute_topo(data, probs, mode):
    """返回 topo 分数 [N]，按 mode 选择映射方式。"""
    num_classes = int(data.y.max().item()) + 1
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
    _, predicted = probs.max(dim=-1)
    pred_cos = cos_all.gather(1, predicted.unsqueeze(1)).squeeze(1)
    mask_other = torch.ones_like(cos_all, dtype=torch.bool)
    mask_other.scatter_(1, predicted.unsqueeze(1), False)
    other_max = cos_all.masked_fill(~mask_other, -1e9).max(dim=1).values
    raw = pred_cos - other_max

    if mode == "original":
        topo = (raw + 2.0) / 4.0
    elif mode == "sigmoid_t0.10":
        topo = torch.sigmoid(raw / 0.10)
    elif mode == "sigmoid_t0.05":
        topo = torch.sigmoid(raw / 0.05)
    elif mode == "linear_x2.5":
        topo = (raw * 2.5 + 0.4).clamp(0.0, 1.0)
    else:
        raise ValueError(mode)
    return topo.clamp(0.0, 1.0)


def train_topo_only(data, seed, mode):
    """Topology-only：α=0, β=1，即 fused = topo。"""
    set_seed(seed)
    classes = int(data.y.max().item()) + 1
    model = SLGAT(data.num_features, HIDDEN, LABEL_HIDDEN, classes, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    start = time.perf_counter()

    # 阶段一
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
            if val > best_val: best_val, best_state, wait = val, clone_state(model), 0
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break
    model.load_state_dict(best_state)
    pretrain_epochs = epoch + 1

    # 阶段二
    best_val, best_state, best_epoch, wait = -1.0, clone_state(model), 0, 0
    best_sel = 0
    for epoch in range(SELF_TRAIN_EPOCHS):
        model.eval()
        with torch.no_grad():
            logits_u = model(data.x, data.edge_index)[0]
            probs = F.softmax(logits_u, -1)
            topo = compute_topo(data, probs, mode)
            selected = (~data.train_mask) & (topo >= THRESHOLD)
            sel_count = int(selected.sum().item())
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
                best_epoch = pretrain_epochs + epoch + 1
                best_sel = sel_count
            else: wait += VALIDATE_EVERY
        if wait >= PATIENCE: break

    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        logits = model(data.x, data.edge_index)[0]
        test_acc = acc(logits, data.y, data.test_mask)
        test_f1 = macro_f1(logits, data.y, data.test_mask, classes)
    return dict(acc=test_acc, f1=test_f1, sel=best_sel, best_ep=best_epoch,
                elapsed=time.perf_counter()-start)


def main():
    print("=" * 96)
    print(f"Quick Check: Topology-only on {DATASET}, 4 mapping modes, δ={THRESHOLD}")
    print(f"Reference: SLGAT (Cora) = {SLGAT_REF}%; Full TA (Cora) = 84.21%")
    print(f"Seeds={len(SEEDS)}; Device={DEVICE}")
    print("=" * 96)

    root = data_root()
    data = Planetoid(root=str(root), name=DATASET)[0].to(DEVICE)
    data = prepare(data)
    n_unlabeled = int((~data.train_mask).sum().item())
    print(f"{DATASET}: nodes={data.num_nodes}, unlabeled={n_unlabeled}")

    # 先快速诊断：用预训练后的 topo 分布看各映射的值域
    print("\n" + "-" * 96)
    print("Pre-check: topo distribution after a short pretrain (1 seed, 50 epochs)")
    set_seed(SEEDS[0])
    m = SLGAT(data.num_features, HIDDEN, LABEL_HIDDEN,
              int(data.y.max().item())+1, DROPOUT).to(DEVICE)
    o = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(50):
        m.train(); o.zero_grad(set_to_none=True)
        lo = m(data.x, data.edge_index)[0]
        F.cross_entropy(lo[data.train_mask], data.y[data.train_mask]).backward(); o.step()
    m.eval()
    with torch.inference_mode():
        probs = F.softmax(m(data.x, data.edge_index)[0], -1)
        for mode in TOPO_MODES:
            t = compute_topo(data, probs, mode)[~data.train_mask]
            q = torch.quantile(t, torch.tensor([0.1,0.5,0.9], device=t.device))
            n_pass = int((t >= THRESHOLD).sum().item())
            print(f"  {mode:<16s} q10={q[0]:.3f} q50={q[1]:.3f} q90={q[2]:.3f} "
                  f"→ pass@{THRESHOLD}={n_pass} ({n_pass/n_unlabeled*100:.1f}%)")
    print("-" * 96)

    # 正式跑 4 种映射 × 10 seed
    results = {}
    for mode in TOPO_MODES:
        print(f"\n{'#'*96}\nTOPO_MODE = {mode}\n{'#'*96}")
        rows = []
        for seed in SEEDS:
            r = train_topo_only(data, seed, mode)
            r["seed"] = seed
            rows.append(r)
            print(f"  seed={seed:>4d} | acc={r['acc']:.4f} f1={r['f1']:.4f} "
                  f"sel={r['sel']:>5d}({r['sel']/n_unlabeled*100:5.1f}%) "
                  f"best_ep={r['best_ep']:>3d}")
        a = np.array([r["acc"] for r in rows])
        f = np.array([r["f1"] for r in rows])
        s = np.array([r["sel"] for r in rows])
        results[mode] = dict(acc_mean=a.mean(), acc_std=a.std(ddof=1),
                             f1_mean=f.mean(), sel_mean=s.mean())
        print(f"  >>> {mode}: Acc={a.mean()*100:.2f}±{a.std(ddof=1)*100:.2f}  "
              f"F1={f.mean()*100:.2f}  Sel={s.mean()/n_unlabeled*100:.1f}%")

    # 汇总对比
    print("\n" + "=" * 96)
    print("COMPARISON (Topology-only, Cora)")
    print("=" * 96)
    print(f"{'Mode':<20s} {'Acc (mean±std)':>18s} {'Sel%':>8s} {'vs SLGAT':>10s} {'vs Full TA':>12s}")
    print("-" * 80)
    for mode in TOPO_MODES:
        r = results[mode]
        delta_slgat = r["acc_mean"]*100 - SLGAT_REF
        delta_full = r["acc_mean"]*100 - 84.21
        print(f"{mode:<20s} {r['acc_mean']*100:>7.2f}±{r['acc_std']*100:<6.2f} "
              f"{r['sel_mean']/n_unlabeled*100:>7.1f}% "
              f"{delta_slgat:>+9.2f} {delta_full:>+11.2f}")
    print(f"\n参照: SLGAT={SLGAT_REF:.2f}, Full TA=84.21")
    print("\n判据：")
    print("  1) Sel% > 20%  → 映射有效，拓扑项能筛出节点")
    print("  2) Acc > 82.55 → Topology-only 优于不筛选，值得保留")
    print("  3) Acc > 84.21 → 拓扑项单独就超过 Full TA（几乎不可能，但若有则重新审视消融）")


if __name__ == "__main__":
    main()
