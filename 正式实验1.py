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
