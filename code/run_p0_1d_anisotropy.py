"""
P0-1d · Embedding 各向异性诊断 (B1 新增建议)
===============================================

Loads per-domain embedding .npy files (from run_stage1_sbar_v2 --save_embeddings),
computes spectral spread to test whether high-anisotropy embeddings (BGE family)
yield lower Vendi Score than low-anisotropy ones (MiniLM).

Metrics per (model, domain):
  - top-k cumulative singular-value ratio (k=1, 5, 10, 20, 50)
  - spectral entropy H = -Σ (σ_i^2 / Σ σ_j^2) · log(σ_i^2 / Σ σ_j^2)
  - participation ratio PR = (Σ σ_i^2)^2 / Σ σ_i^4  (effective rank proxy)

Also aggregates: per model, mean/median across 8 domains.

Output:
  anisotropy_results.csv
  anisotropy_top10_ratio.png

Usage:
  python run_p0_1d_anisotropy.py \
    --emb_dirs minilm:results_minilm bge_small:results_bge_small bge_large:results_bge_large legal:results_legal \
    --output_dir anisotropy_out \
    --pos head
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

DOMAINS = ["Pile-CC", "Wikipedia_en", "ArXiv", "Github",
           "PubMed_Central", "FreeLaw", "StackExchange", "USPTO_Backgrounds"]


def analyze(X, topk_list=(1, 5, 10, 20, 50), mode="l2norm"):
    """Compute singular spectrum.

    Modes (B1 第 5 点反馈 ①):
      - "l2norm"  : L2-normalize each row (default; probes directional concentration)
      - "raw"     : no preprocessing (probes full anisotropy incl. origin bias)
      - "centered": subtract mean (probes net multi-diversity after de-biasing)
    """
    if mode == "l2norm":
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        Xn = X / norm
    elif mode == "centered":
        Xn = X - X.mean(axis=0)
    elif mode == "raw":
        Xn = X
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # SVD (truncated to min(n, d) is automatic with full_matrices=False)
    _, s, _ = np.linalg.svd(Xn, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total <= 0:
        return {}
    p = s2 / total
    H = -np.sum(p * np.log(p + 1e-12))
    PR = (s2.sum() ** 2) / (s2 ** 2).sum()
    d = len(s)

    out = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "mode": mode,
        "spectrum_entropy": float(H),
        "participation_ratio": float(PR),
        "normalized_PR": float(PR / d),  # 1.0 = isotropic, ~0 = strongly anisotropic
    }
    for k in topk_list:
        k_eff = min(k, d)
        out[f"top{k}_ratio"] = float(s2[:k_eff].sum() / total)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dirs", nargs="+", required=True,
                        help="model_tag:dir pairs, e.g. minilm:results_minilm bge_small:results_bge_small")
    parser.add_argument("--output_dir", default="anisotropy_out")
    parser.add_argument("--pos", default="head")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    MODES = ["l2norm", "raw", "centered"]  # B1 ① 三份都存
    for item in args.emb_dirs:
        if ":" not in item:
            log.error(f"bad --emb_dirs item {item} (want tag:dir)")
            continue
        tag, path = item.split(":", 1)
        emb_dir = Path(path)
        for dom in DOMAINS:
            npy = emb_dir / f"embeddings_{dom}_{args.pos}.npy"
            if not npy.exists():
                log.warning(f"[MISS] {npy}")
                continue
            X = np.load(npy)
            for mode in MODES:
                m = analyze(X, mode=mode)
                m.update({"model": tag, "domain": dom})
                log.info(f"[{tag}] {dom} [{mode:>8}]: top10={m.get('top10_ratio', np.nan):.3f}  H={m.get('spectrum_entropy', np.nan):.3f}  PR_norm={m.get('normalized_PR', np.nan):.3f}")
                rows.append(m)

    if not rows:
        log.error("no embedding files found; nothing to do")
        return

    df = pd.DataFrame(rows)
    csv = out_dir / "anisotropy_results.csv"
    df.to_csv(csv, index=False)
    log.info(f"saved {csv}")

    # Per-(model, mode) summary
    log.info("===== Per-(model, mode) mean top-10 ratio (across domains) =====")
    summary = df.groupby(["model", "mode"])[["top1_ratio", "top5_ratio", "top10_ratio", "top20_ratio",
                                              "spectrum_entropy", "normalized_PR"]].mean()
    log.info(f"\n{summary.to_string()}")
    summary.to_csv(out_dir / "anisotropy_model_summary.csv")

    # B1 ① diagnostic: raw - centered gap is the "embedding bias" signature
    log.info("===== Raw vs Centered gap (embedding bias diagnostic, B1 ①) =====")
    diag_rows = []
    for (m_tag, dom), sub in df.groupby(["model", "domain"]):
        row = {"model": m_tag, "domain": dom}
        for mode in ["raw", "centered", "l2norm"]:
            sub2 = sub[sub["mode"] == mode]
            if len(sub2) == 0:
                continue
            row[f"{mode}_top10"] = float(sub2["top10_ratio"].iloc[0])
            row[f"{mode}_PR_norm"] = float(sub2["normalized_PR"].iloc[0])
        row["raw_minus_centered_top10"] = row.get("raw_top10", np.nan) - row.get("centered_top10", np.nan)
        diag_rows.append(row)
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir / "anisotropy_raw_vs_centered.csv", index=False)
    log.info(f"saved {out_dir / 'anisotropy_raw_vs_centered.csv'}")

    # Plot: l2norm mode only (main diagnostic)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        sub_l2 = df[df["mode"] == "l2norm"]
        models = sub_l2["model"].unique()
        x = np.arange(len(DOMAINS))
        width = 0.8 / max(1, len(models))
        for i, m in enumerate(models):
            sub = sub_l2[sub_l2["model"] == m].set_index("domain").reindex(DOMAINS)
            plt.bar(x + i * width, sub["top10_ratio"].values, width, label=m)
        plt.xticks(x + (len(models) - 1) * width / 2, DOMAINS, rotation=45, ha="right")
        plt.ylabel("top-10 singular^2 / total  (l2norm)")
        plt.title(f"Anisotropy top-10 ratio (pos={args.pos}, mode=l2norm)")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "anisotropy_top10_ratio.png", dpi=150)

        # Second plot: raw_top10 - centered_top10 heatmap-style bars
        plt.figure(figsize=(10, 5))
        for i, m in enumerate(models):
            sub = diag_df[diag_df["model"] == m].set_index("domain").reindex(DOMAINS)
            plt.bar(x + i * width, sub["raw_minus_centered_top10"].values, width, label=m)
        plt.xticks(x + (len(models) - 1) * width / 2, DOMAINS, rotation=45, ha="right")
        plt.ylabel("raw_top10 − centered_top10  (embedding bias)")
        plt.title("Raw vs Centered top-10 gap (high gap = strong embedding offset)")
        plt.legend(); plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "anisotropy_raw_vs_centered.png", dpi=150)
        log.info(f"saved plots in {out_dir}")
    except Exception as e:
        log.error(f"plot failed: {e}")


if __name__ == "__main__":
    main()
