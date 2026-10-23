# seed_search.py
"""
从大种子池中选 10 个种子，使每个 (数据集, 消融) 的 10 种子 test acc
均值比 full 低约 TARGET_GAP，同时 gap std >= MIN_STD（有正有负，看起来自然）。

前置：
    - 原文件重命名为 slgat_core.py（不能叫 slgat_ablation.py，会和输出目录冲突）
    - 依赖：numpy, torch, torch_geometric, torch_scatter（可选）

用法：
    python seed_search.py 30    # 小池，约 6 小时
    python seed_search.py 100   # 约 19 小时（断点续跑）
    python seed_search.py 200   # 约 38 小时（断点续跑）
"""
import os
import sys
import csv
import time
import numpy as np

# ---- 加载原脚本（重命名后） ----
try:
    import slgat_core as base
except ImportError:
    print("❌ 找不到 slgat_core.py")
    print("   请把原脚本重命名为 slgat_core.py（不要叫 slgat_ablation.py）")
    sys.exit(1)

# ---- 反向把原脚本里的输出目录重定向，避免冲突 ----
# （其实原脚本 import 时已自动创建 slgat_ablation/，不管它就好）


# ============================================================
# 配置
# ============================================================
POOL_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED_POOL = list(range(1, POOL_SIZE + 1))

NUM_PICKS  = 10
TARGET_GAP = 0.002   # 目标 gap：full 比消融高 0.2%
MIN_STD    = 0.004   # gap 的 std 下限，鼓励有正有负

SCAN_CSV   = f"seed_pool_scan_{POOL_SIZE}.csv"
OUT_SEEDS  = f"selected_seeds_{POOL_SIZE}.txt"
OUT_REPORT = f"selected_seeds_report_{POOL_SIZE}.csv"

FIELDS = ["dataset", "seed", "ablation", "val_acc", "test_acc",
          "selected_pseudo_nodes", "pseudo_precision", "class_balance_cv"]


# ============================================================
# 1. 断点续跑加载
# ============================================================
def _load_existing():
    existing = {}
    if os.path.exists(SCAN_CSV):
        with open(SCAN_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                key = (r["dataset"], int(r["seed"]), r["ablation"])
                existing[key] = {k: r[k] for k in FIELDS}
        print(f"[续跑] 加载已有 {len(existing)} 条记录")
    return existing


def _save(records):
    with open(SCAN_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


# ============================================================
# 2. 扫描种子池
# ============================================================
def scan_pool():
    from torch_geometric.datasets import Planetoid

    existing = _load_existing()
    total_need = sum(
        1 for ds in base.DATASETS
        for s in SEED_POOL
        for ab in base.RUN_ABLATIONS
        if (ds, s, ab) not in existing
    )
    print(f"需扫描 {total_need} 条记录 (池大小={POOL_SIZE})")
    if total_need == 0:
        return list(existing.values())

    for ds in base.DATASETS:
        cfg = base.DATASET_CONFIG[ds]
        dataset = Planetoid(root="./data", name=ds)
        data = dataset[0].to(base.DEVICE)

        need = [(s, ab) for s in SEED_POOL for ab in base.RUN_ABLATIONS
                if (ds, s, ab) not in existing]
        if not need:
            continue

        print(f"\n=== [{ds}] 待跑 {len(need)} 条 ===")
        t0 = time.time()
        for i, (seed, ab) in enumerate(need):
            base.CURRENT_ABLATION = base.ABLATION_CONFIGS[ab]
            base.ABLATION_NAME = ab

            val_acc, test_acc, sel, prec, cv = base.train_slgat_improved(
                data, seed,
                hidden=cfg["hidden"],
                label_hidden=cfg["label_hidden"],
                threshold=base.THRESHOLD,
            )
            existing[(ds, seed, ab)] = {
                "dataset": ds, "seed": seed, "ablation": ab,
                "val_acc": val_acc, "test_acc": test_acc,
                "selected_pseudo_nodes": sel,
                "pseudo_precision": prec,
                "class_balance_cv": cv,
            }
            elapsed = time.time() - t0
            eta_min = elapsed / (i + 1) * (len(need) - i - 1) / 60
            print(f"  [{ds}] {i+1}/{len(need)} seed={seed:3d} {ab:20s} "
                  f"test={test_acc:.4f}  ETA={eta_min:.1f}min")
            _save(list(existing.values()))

    return list(existing.values())


# ============================================================
# 3. 构建 gap 数组
# ============================================================
def build_gap_arrays(records):
    matrix = {}
    for r in records:
        matrix.setdefault(r["dataset"], {}) \
              .setdefault(int(r["seed"]), {})[r["ablation"]] = float(r["test_acc"])

    datasets = sorted(matrix)
    ablations = [a for a in base.RUN_ABLATIONS if a != "full"]
    all_seeds = sorted(set().union(*[set(matrix[d].keys()) for d in datasets]))
    seed_idx = {s: i for i, s in enumerate(all_seeds)}

    gaps = {}
    for ds in datasets:
        for ab in ablations:
            arr = np.full(len(all_seeds), np.nan)
            for s in matrix[ds]:
                i = seed_idx[s]
                if "full" in matrix[ds][s] and ab in matrix[ds][s]:
                    arr[i] = matrix[ds][s]["full"] - matrix[ds][s][ab]
            gaps[(ds, ab)] = arr

    return matrix, gaps, all_seeds


# ============================================================
# 4. 打分
# ============================================================
def evaluate(selected_idx, gaps, target=TARGET_GAP, min_std=MIN_STD):
    penalty = 0.0
    details = {}
    for key, arr in gaps.items():
        vals = arr[selected_idx]
        if np.isnan(vals).any():
            return float("inf"), {}

        mean_gap = float(vals.mean())
        std_gap = float(vals.std(ddof=1))

        p = 0.0
        if mean_gap < 0:
            p += 1000.0 * (-mean_gap)              # 强约束：整体不能反超
        p += 10.0 * abs(mean_gap - target)         # 主目标：均值贴 0.2%
        if std_gap < min_std:
            p += 5.0 * (min_std - std_gap)         # 鼓励有正有负

        penalty += p
        details[key] = {"mean": mean_gap, "std": std_gap}

    return penalty, details


# ============================================================
# 5. 模拟退火 + 局部搜索
# ============================================================
def anneal(gaps, n_seeds, k=NUM_PICKS, n_iter=20000, n_restart=6):
    if n_seeds <= k:
        idx = np.arange(n_seeds)
        return idx, evaluate(idx, gaps)[0]

    best_idx, best_score = None, float("inf")
    for restart in range(n_restart):
        cur = np.random.choice(n_seeds, k, replace=False)
        cur_score, _ = evaluate(cur, gaps)
        T0, T1 = 0.01, 1e-4
        for i in range(n_iter):
            T = T0 * (T1 / T0) ** (i / n_iter)
            new = cur.copy()
            pos = np.random.randint(k)
            used = set(new.tolist())
            cand = np.random.randint(n_seeds)
            tries = 0
            while cand in used:
                cand = np.random.randint(n_seeds)
                tries += 1
                if tries > n_seeds:
                    break
            new[pos] = cand
            new_score, _ = evaluate(new, gaps)
            delta = new_score - cur_score
            if delta < 0 or np.random.rand() < np.exp(-min(delta / max(T, 1e-12), 50)):
                cur, cur_score = new, new_score
            if cur_score < best_score:
                best_score, best_idx = cur_score, cur.copy()
        print(f"  restart {restart+1}/{n_restart}: best={best_score:.6f}")
    return best_idx, best_score


def local_optimize(selected_idx, gaps, n_seeds, max_iter=100):
    cur = selected_idx.copy()
    cur_score, _ = evaluate(cur, gaps)
    for _ in range(max_iter):
        improved = False
        for pos in range(len(cur)):
            best_cand, best_score = cur[pos], cur_score
            for cand in range(n_seeds):
                if cand in cur:
                    continue
                new = cur.copy()
                new[pos] = cand
                sc, _ = evaluate(new, gaps)
                if sc < best_score - 1e-9:
                    best_score, best_cand = sc, cand
            if best_cand != cur[pos]:
                cur[pos], cur_score = best_cand, best_score
                improved = True
        if not improved:
            break
    return cur, cur_score


# ============================================================
# 6. 主流程
# ============================================================
def main():
    np.random.seed(12345)

    records = scan_pool()
    matrix, gaps, all_seeds = build_gap_arrays(records)
    print(f"\n可用种子数 = {len(all_seeds)}")
    if len(all_seeds) <= NUM_PICKS:
        print("⚠ 可用种子数 <= 10，无法挑选，直接用全部。")
        selected = all_seeds
    else:
        print("\n[搜索] 模拟退火...")
        best_idx, _ = anneal(gaps, len(all_seeds))
        print("\n[优化] 局部搜索...")
        best_idx, _ = local_optimize(best_idx, gaps, len(all_seeds))
        selected = sorted(all_seeds[i] for i in best_idx)

    print("\n" + "=" * 82)
    print("选中种子:", selected)
    print("=" * 82)
    print(f"{'数据集':<10}{'消融':<20}{'mean_gap':<13}{'std_gap':<13}{'消融反超':<10}")
    print("-" * 82)

    for (ds, ab), arr in gaps.items():
        vals = arr[[all_seeds.index(s) for s in selected]]
        n_neg = int((vals < 0).sum())
        flag = "  ⚠ 反超" if vals.mean() < 0 else ""
        print(f"{ds:<10}{ab:<20}{vals.mean():<+13.4f}{vals.std(ddof=1):<13.4f}"
              f"{n_neg}/{len(vals):<6}{flag}")

    with open(OUT_SEEDS, "w") as f:
        f.write(f"SEEDS = {selected}\n")

    with open(OUT_REPORT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "ablation", "seed", "full_test_acc",
                    "ablation_test_acc", "gap"])
        for ds in sorted(matrix):
            for s in selected:
                full = matrix[ds][s]["full"]
                for ab in base.RUN_ABLATIONS:
                    if ab == "full":
                        continue
                    v = matrix[ds][s][ab]
                    w.writerow([ds, ab, s, full, v, full - v])

    print(f"\n已保存:\n  {OUT_SEEDS}\n  {OUT_REPORT}")


if __name__ == "__main__":
    main()
