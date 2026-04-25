"""
P0-2 · K-means K scan + seed scan (B1 #4 建议合并)
=====================================================

Reads pre-computed MiniLM embeddings (.npy from run_stage1_sbar_v2 --save_embeddings),
scans K and random_state for silhouette on 3 target subsets:
  FreeLaw / StackExchange / PubMed Central

K grid:        {2, 5, 10, 20, 50}
Seed grid:     {1, 2, 3, 42, 100}  (at fixed K=5)

Output:
  results_kscan/kscan_results.csv
  results_kscan/kscan_curves.png
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

K_GRID = [2, 5, 10, 20, 50]
SEED_GRID = [1, 2, 3, 42, 100]
DEFAULT_DOMAINS = ["FreeLaw", "StackExchange", "PubMed_Central"]
DEFAULT_EMB_DIR = "results_minilm"  # where embeddings_{domain}_head.npy lives


def silhouette(X, k, seed, sample_size=5000):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(X)
    if n < k + 1:
        return float("nan")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    sub = min(sample_size, n)
    return float(silhouette_score(X, labels, sample_size=sub, random_state=seed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default=DEFAULT_EMB_DIR,
                        help="Dir containing embeddings_{domain}_head.npy")
    parser.add_argument("--output_dir", default="results_kscan")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--pos", default="head")
    args = parser.parse_args()

    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for domain in args.domains:
        npy = emb_dir / f"embeddings_{domain}_{args.pos}.npy"
        if not npy.exists():
            log.error(f"[MISS] {npy}")
            continue
        X = np.load(npy)
        log.info(f"[{domain}] loaded {X.shape} from {npy}")

        # K scan @ seed=42
        for k in K_GRID:
            s = silhouette(X, k, seed=42)
            log.info(f"[{domain}] K={k:<3} seed=42  silhouette={s:.4f}")
            rows.append({"domain": domain, "mode": "kscan", "K": k, "seed": 42, "silhouette": s})

        # Seed scan @ K=5
        for seed in SEED_GRID:
            s = silhouette(X, 5, seed=seed)
            log.info(f"[{domain}] K=5   seed={seed:<3} silhouette={s:.4f}")
            rows.append({"domain": domain, "mode": "seedscan", "K": 5, "seed": seed, "silhouette": s})

    df = pd.DataFrame(rows)
    csv = out_dir / "kscan_results.csv"
    df.to_csv(csv, index=False)
    log.info(f"saved {csv}")

    # Plot K-silhouette curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        for domain in df["domain"].unique():
            sub = df[(df["domain"] == domain) & (df["mode"] == "kscan")].sort_values("K")
            plt.plot(sub["K"], sub["silhouette"], marker="o", label=domain)
        plt.xscale("log")
        plt.xlabel("K (log)"); plt.ylabel("silhouette")
        plt.title("K scan @ seed=42")
        plt.legend(); plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        for domain in df["domain"].unique():
            sub = df[(df["domain"] == domain) & (df["mode"] == "seedscan")].sort_values("seed")
            plt.plot(sub["seed"].astype(str), sub["silhouette"], marker="s", label=domain)
        plt.xlabel("seed"); plt.ylabel("silhouette")
        plt.title("Seed scan @ K=5")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "kscan_curves.png", dpi=150)
        log.info(f"saved {out_dir / 'kscan_curves.png'}")

        # Stats summary
        log.info("===== Seed-scan stability (at K=5) =====")
        for domain in df["domain"].unique():
            sub = df[(df["domain"] == domain) & (df["mode"] == "seedscan")]["silhouette"]
            log.info(f"  {domain}: mean={sub.mean():.4f}  std={sub.std():.4f}  range=[{sub.min():.4f}, {sub.max():.4f}]")

        log.info("===== K-scan extremes =====")
        for domain in df["domain"].unique():
            sub = df[(df["domain"] == domain) & (df["mode"] == "kscan")]
            amax = sub.loc[sub["silhouette"].idxmax()]
            log.info(f"  {domain}: argmax silhouette at K={int(amax['K'])} → {amax['silhouette']:.4f}")
    except Exception as e:
        log.error(f"plot/summary failed: {e}")


if __name__ == "__main__":
    main()
