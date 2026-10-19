model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
    return best_val, test_acc, selected_count


# ============================================================
# 单个数据集的完整实验
# ============================================================
def run_one_dataset(dataset_name):
    cfg = DATASET_CONFIG[dataset_name]
    hidden = cfg["hidden"]
    label_hidden = cfg["label_hidden"]

    out_dir = os.path.join(ROOT_OUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 80)
    print("# DATASET:", dataset_name)
    print("# config: hidden=%d, label_hidden=%d" % (hidden, label_hidden))
    print("#" * 80)

    dataset = Planetoid(root="./data", name=dataset_name)
    data = dataset[0].to(DEVICE)

    results = []
    t_dataset_start = time.time()

    for seed in SEEDS:
        print("\n--- [%s] seed=%d ---" % (dataset_name, seed))
        t0 = time.time()
        
        # 直接运行你的改进版模型
        val_acc, test_acc, sel = train_slgat_improved(
            data, seed, 
            hidden=hidden, label_hidden=label_hidden,
            threshold=THRESHOLD
        )
        sel_str = str(sel)
        
        dt = time.time() - t0
        print("  %-24s | Val=%.4f | Test=%.4f | Sel=%-6s | %.1fs" % (
            "SLGAT + improved", val_acc, test_acc, sel_str, dt
        ))
        results.append({
            "seed": seed,
            "method": "SLGAT + improved",
            "val_acc": val_acc,
            "test_acc": test_acc,
            "selected_pseudo_nodes": sel_str,
        })

    total_min = (time.time() - t_dataset_start) / 60.0
    print("\n[%s] total time: %.1f min" % (dataset_name, total_min))

    # 保存结果
    per_run_path = os.path.join(out_dir, "results_per_run.csv")
    with open(per_run_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "method", "val_acc", "test_acc", "selected_pseudo_nodes"
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_rows = []
    val_values = np.array([r["val_acc"] for r in results])
    test_values = np.array([r["test_acc"] for r in results])
    
    print("\n" + "=" * 80)
    print("SUMMARY [%s] (%d seeds)" % (dataset_name, len(SEEDS)))
    print("=" * 80)
    print("  Val  acc: %.4f +/- %.4f" % (val_values.mean(), val_values.std(ddof=1)))
    print("  Test acc: %.4f +/- %.4f" % (test_values.mean(), test_values.std(ddof=1)))
    
    summary_rows.append({
        "dataset": dataset_name,
        "method": "SLGAT + improved",
        "val_mean": val_values.mean(),
        "val_std": val_values.std(ddof=1),
        "test_mean": test_values.mean(),
        "test_std": test_values.std(ddof=1),
    })

    summary_path = os.path.join(out_dir, "results_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "val_mean", "val_std", "test_mean", "test_std"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n[%s] per-run saved to: %s" % (dataset_name, per_run_path))
    print("[%s] summary saved to: %s" % (dataset_name, summary_path))

    return summary_rows


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 80)
    print("SLGAT + Improved Topology Filtering")
    print("Datasets:", DATASETS)
    print("Seeds:", SEEDS)
    print("device:", DEVICE)
    print("=" * 80)

    t_all_start = time.time()
    all_summary = []
    for ds in DATASETS:
        rows = run_one_dataset(ds)
        all_summary.extend(rows)

    # 跨数据集总汇总
    all_summary_path = os.path.join(ROOT_OUT_DIR, "all_datasets_summary.csv")
    with open(all_summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "method", "val_mean", "val_std", "test_mean", "test_std"
        ])
        writer.writeheader()
        writer.writerows(all_summary)

    total_min = (time.time() - t_all_start) / 60.0
    print("\n" + "=" * 80)
    print("ALL DONE. Total time: %.1f min" % total_min)
    print("All summary saved to:", all_summary_path)
