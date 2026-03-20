#!/usr/bin/env python3
"""
s16_run_all.py - Complete S1.6 pipeline: pretrain + extract + 4-way compare + diagnostics.

Runs everything in sequence to minimize manual steps.

How to run:
  cd project
  OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3.14 scripts/s16_run_all.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_device(pref="auto"):
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )
    logger = logging.getLogger("s16")

    with open(PROJECT_ROOT / "config" / "s16_config.yaml") as f:
        cfg = yaml.safe_load(f)

    s0_dir = PROJECT_ROOT / cfg["paths"]["s0_dir"]
    s1_dir = PROJECT_ROOT / cfg["paths"]["s1_dir"]
    s15_dir = PROJECT_ROOT / cfg["paths"]["s15_dir"]
    s16_dir = PROJECT_ROOT / cfg["paths"]["s16_dir"]
    s16_dir.mkdir(parents=True, exist_ok=True)
    (s16_dir / "checkpoints").mkdir(exist_ok=True)

    device = get_device(cfg.get("runtime", {}).get("device", "auto"))
    enc = cfg["encoder"]
    con = cfg["contrastive"]
    pt = cfg["pretraining"]

    # ============================================================
    # Phase 1: S1.6 Pretraining (lambda=0.2)
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: S1.6 Pretraining (max_lambda=0.2)")
    logger.info("=" * 60)

    from s15.pretrain_contrastive import pretrain_contrastive

    pretrain_log = pretrain_contrastive(
        s0_dir=s0_dir, output_dir=s16_dir,
        n_features=enc["n_features"], d_model=enc["d_model"],
        n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        d_ff=enc["d_ff"], dropout=enc["dropout"],
        view_len=con["view_len"], mask_ratio=con["mask_ratio"],
        temperature=con["temperature"], proj_dim=con["proj_dim"],
        max_lambda=con["max_lambda"], warmup_epochs=con["warmup_epochs"],
        epochs=pt["epochs"], batch_size=pt["batch_size"],
        lr=pt["lr"], weight_decay=pt["weight_decay"],
        patience=pt["patience"], grad_clip=pt["grad_clip"],
        device=device, seed=pt["seed"],
    )

    # ============================================================
    # Phase 2: Extract S1.6 Embeddings
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Extract S1.6 Embeddings")
    logger.info("=" * 60)

    import torch
    from s1.encoder import ICUTransformerEncoder

    ckpt = torch.load(s16_dir / "checkpoints" / "pretrain_best.pt",
                       map_location=device, weights_only=True)
    ec = ckpt["config"]

    encoder = ICUTransformerEncoder(
        n_features=ec["n_features"], d_model=ec["d_model"],
        n_heads=ec["n_heads"], n_layers=ec["n_layers"],
        d_ff=ec["d_ff"], dropout=0.0,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.train(False)

    continuous = np.load(s0_dir / "processed" / "continuous.npy")
    masks = np.load(s0_dir / "processed" / "masks_continuous.npy")

    all_emb = []
    with torch.no_grad():
        for start in range(0, len(continuous), 128):
            end = min(start + 128, len(continuous))
            x = torch.from_numpy(continuous[start:end]).float().to(device)
            m = torch.from_numpy(masks[start:end]).float().to(device)
            all_emb.append(encoder(x, m).cpu().numpy())

    emb_s16 = np.concatenate(all_emb)
    np.save(s16_dir / "embeddings_s16.npy", emb_s16)
    logger.info(f"S1.6 embeddings: {emb_s16.shape}")

    # ============================================================
    # Phase 3: 4-Way Clustering Comparison
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: 4-Way Clustering Comparison")
    logger.info("=" * 60)

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    static = pd.read_csv(s0_dir / "static.csv")
    mortality = static["mortality_inhospital"].fillna(0).values
    center_a = (static["center_id"] == "center_a").values
    center_b = (static["center_id"] == "center_b").values

    SEEDS = [42, 123, 456, 789, 2024]

    methods = {
        "PCA": np.load(s1_dir / "embeddings_pca.npy"),
        "S1_masked": np.load(s1_dir / "embeddings_ss.npy"),
        "S15_lam05": np.load(s15_dir / "embeddings_s15.npy"),
        "S16_lam02": emb_s16,
    }

    comparison = {}
    for method_name, emb in methods.items():
        method_res = {}
        for k in [2, 4]:
            seed_metrics = []
            for seed in SEEDS:
                km = KMeans(n_clusters=k, n_init=10, random_state=seed, max_iter=300)
                labels = km.fit_predict(emb)

                sil = silhouette_score(emb, labels)
                morts = [float(mortality[labels == c].mean()) if (labels == c).sum() > 0 else 0.0 for c in range(k)]
                dist_a = np.bincount(labels[center_a], minlength=k) / center_a.sum()
                dist_b = np.bincount(labels[center_b], minlength=k) / center_b.sum()

                seed_metrics.append({
                    "sil": sil, "mort_range": max(morts) - min(morts),
                    "mort_min": min(morts), "mort_max": max(morts),
                    "center_l1": float(np.abs(dist_a - dist_b).sum()),
                })

            agg = {}
            for key in ["sil", "mort_range", "mort_min", "mort_max", "center_l1"]:
                vals = [m[key] for m in seed_metrics]
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            method_res[f"K={k}"] = agg

        comparison[method_name] = method_res

    # Print table
    for k in [2, 4]:
        print(f"\n{'='*90}")
        print(f"  K={k} Clustering (mean±std, 5 seeds)")
        print(f"{'='*90}")
        print(f"  {'Metric':<20s}", end="")
        for m in methods:
            print(f" {m:>16s}", end="")
        print()
        print(f"  {'-'*88}")
        for label, key in [("Silhouette", "sil"), ("Mort range", "mort_range"),
                           ("Mort min", "mort_min"), ("Mort max", "mort_max"),
                           ("Center L1 ↓", "center_l1")]:
            print(f"  {label:<20s}", end="")
            for m in methods:
                a = comparison[m][f"K={k}"][key]
                if "mort" in key:
                    print(f" {a['mean']:>7.1%}±{a['std']:>4.1%} ", end="")
                else:
                    print(f" {a['mean']:>7.4f}±{a['std']:>4.4f} ", end="")
            print()
    print(f"{'='*90}")

    with open(s16_dir / "comparison_4way.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # ============================================================
    # Phase 4: Full Diagnostics (all 4, with corrected center probe)
    # ============================================================
    logger.info("=" * 60)
    logger.info("PHASE 4: Diagnostics (corrected center probe)")
    logger.info("=" * 60)

    from s15.diagnostics import run_all_diagnostics, print_diagnostics_comparison

    static_path = s0_dir / "static.csv"
    splits_path = s0_dir / "splits.json"
    masks_path = s0_dir / "processed" / "masks_continuous.npy"

    diag_reports = []
    for method_name, emb_path in [
        ("PCA", s1_dir / "embeddings_pca.npy"),
        ("S1_masked", s1_dir / "embeddings_ss.npy"),
        ("S15_lam05", s15_dir / "embeddings_s15.npy"),
        ("S16_lam02", s16_dir / "embeddings_s16.npy"),
    ]:
        report = run_all_diagnostics(
            np.load(emb_path), static_path, splits_path, masks_path,
            s16_dir / f"diagnostics_{method_name}.json", label=method_name,
        )
        diag_reports.append(report)

    print_diagnostics_comparison(diag_reports)

    logger.info("=" * 60)
    logger.info("S1.6 COMPLETE. All results in data/s16/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
