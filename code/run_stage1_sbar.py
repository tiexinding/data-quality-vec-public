"""
阶梯 1 · s̄ 四分量 MVP
====================

Stage 1 MVP for the s̄ (s-bar) 4-component data-quality framework.
Computes 4 quantities per Pile subset:
  - s̄_con (concentration): K-means silhouette score
  - s̄_num (effective count): MinHash LSH unique fraction
  - s̄_rep (repetition): 1 - s̄_num
  - s̄_div (distribution entropy): Vendi Score

Run:
    pip install -r requirements.txt
    python smoke_test.py               # 3-5 min verification
    python run_stage1_sbar.py          # 1-2 day full run

Output:
    results/sbar_raw.csv
    results/sbar_correlation_matrix.png
    results/sbar_by_domain.png
    results/sbar_details.json

Author: NeuralCAE / B2
Date: 2026-04-23
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Default Pile subsets (monology/pile-uncopyrighted). Any that fail to
# load are skipped with a warning.
# ----------------------------------------------------------------------
DEFAULT_SUBSETS = [
    "Pile-CC",
    "Wikipedia (en)",
    "Books3",
    "ArXiv",
    "Github",
    "PubMed Central",
    "FreeLaw",
    "StackExchange",
    "OpenWebText2",
    "USPTO Backgrounds",
]


# ----------------------------------------------------------------------
# Data loading (streaming + on-disk cache of sampled texts)
# ----------------------------------------------------------------------
def _safe(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_and_sample(subset_name, n_samples, cache_dir, dataset_id="monology/pile-uncopyrighted",
                     max_scan=200000, log_every=10000):
    """Stream the Pile dataset, filter to one subset, take first n_samples non-empty texts.

    Guards added to avoid infinite hangs on subsets removed from the uncopyrighted version
    (e.g. Books3): if after scanning `max_scan` rows we still don't have n_samples, we
    give up and warn. Also logs progress every `log_every` rows so hung streams are visible.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    cache_file = cache_dir / f"texts_{_safe(subset_name)}_{n_samples}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = json.load(f)
        log.info(f"[cache] texts hit for {subset_name}: {len(texts)} samples")
        return texts

    log.info(f"[load] streaming {dataset_id}, filter subset='{subset_name}', target n={n_samples}, max_scan={max_scan}")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    texts = []
    pbar = tqdm(total=n_samples, desc=f"sample {subset_name}")
    scanned = 0
    last_logged = 0
    for ex in ds:
        scanned += 1
        # Heartbeat every log_every rows
        if scanned - last_logged >= log_every:
            log.info(f"[scan] {subset_name}: scanned={scanned}, matched={len(texts)}")
            last_logged = scanned
        # Hard cap
        if scanned >= max_scan:
            log.warning(f"[skip] {subset_name}: scanned {scanned} rows but only {len(texts)} matched. "
                        f"Likely subset removed from {dataset_id}. Giving up.")
            break

        meta = ex.get("meta", {})
        if isinstance(meta, dict):
            label = meta.get("pile_set_name") or meta.get("subset") or meta.get("name")
        else:
            label = str(meta)
        if label == subset_name:
            text = ex.get("text") or ex.get("content") or ""
            text = text.strip()
            if len(text) >= 50:
                texts.append(text)
                pbar.update(1)
                if len(texts) >= n_samples:
                    break
    pbar.close()

    log.info(f"[load] got {len(texts)} samples for {subset_name} (scanned {scanned} rows)")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    return texts


# ----------------------------------------------------------------------
# Embedding (sentence-BERT)
# ----------------------------------------------------------------------
def compute_embeddings(texts, model, batch_size=64):
    # Truncate each text to avoid huge tokenization; model will auto-truncate to 512
    # but we limit to 2000 chars to save tokenization time
    texts_trunc = [t[:2000] for t in texts]
    log.info(f"[embed] encoding {len(texts_trunc)} texts, batch_size={batch_size}")
    emb = model.encode(
        texts_trunc,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep raw; we normalize in s̄_div
    )
    return np.asarray(emb, dtype=np.float32)


# ----------------------------------------------------------------------
# s̄_con: K-means silhouette
# ----------------------------------------------------------------------
def compute_sbar_con(embeddings, k=5, sample_size=5000):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n < k + 1:
        return float("nan")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    sub = min(sample_size, n)
    score = silhouette_score(embeddings, labels, sample_size=sub, random_state=42)
    return float(score)


# ----------------------------------------------------------------------
# s̄_num: MinHash LSH unique fraction
# ----------------------------------------------------------------------
def compute_sbar_num(texts, threshold=0.8, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    from tqdm import tqdm

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_count = 0
    for i, text in enumerate(tqdm(texts, desc="MinHash LSH")):
        m = MinHash(num_perm=num_perm)
        # shingle by word; convert to bytes
        tokens = text.split()
        if not tokens:
            continue
        for tok in tokens:
            m.update(tok.encode("utf-8"))
        key = f"d{i}"
        if not lsh.query(m):
            lsh.insert(key, m)
            unique_count += 1
    if len(texts) == 0:
        return float("nan")
    return unique_count / len(texts)


# ----------------------------------------------------------------------
# s̄_div: Vendi Score (cosine kernel spectrum entropy)
# Normalized by N so comparable across subsets.
# ----------------------------------------------------------------------
def compute_sbar_div(embeddings, sub_n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    if n == 0:
        return float("nan")
    # Subsample if too large (eigendecomposition is O(n^3))
    if n > sub_n:
        idx = rng.choice(n, sub_n, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    # L2 normalize
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    emb = emb / norm
    # Cosine similarity matrix, then K / n
    K = emb @ emb.T
    K = K / len(emb)
    # symmetric eigendecomp
    eig = np.linalg.eigvalsh(K)
    eig = np.clip(eig, 0.0, None)
    s = eig.sum()
    if s <= 0:
        return float("nan")
    p = eig / s
    # Shannon entropy
    H = -np.sum(p * np.log(p + 1e-12))
    vendi = float(np.exp(H))
    # Normalize by sample size so it's comparable across subset samples
    return vendi / len(emb)


# ----------------------------------------------------------------------
# Per-subset pipeline (with result cache)
# ----------------------------------------------------------------------
def process_subset(subset_name, n_samples, model, cache_dir):
    result_file = cache_dir / f"sbar_{_safe(subset_name)}.json"
    if result_file.exists():
        with open(result_file) as f:
            r = json.load(f)
        log.info(f"[cache] result hit for {subset_name}: {r}")
        return r

    texts = load_and_sample(subset_name, n_samples, cache_dir)
    if len(texts) < 50:
        log.warning(f"[skip] {subset_name}: only {len(texts)} samples, skipping")
        return None

    embeddings = compute_embeddings(texts, model)

    r = {
        "subset": subset_name,
        "n": len(texts),
        "s_bar_con": compute_sbar_con(embeddings),
        "s_bar_num": compute_sbar_num(texts),
        "s_bar_div": compute_sbar_div(embeddings),
    }
    r["s_bar_rep"] = 1.0 - r["s_bar_num"] if not np.isnan(r["s_bar_num"]) else float("nan")

    with open(result_file, "w") as f:
        json.dump(r, f, indent=2)
    log.info(f"[done] {subset_name} -> {r}")
    return r


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def make_plots(df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    s_cols = ["s_bar_con", "s_bar_num", "s_bar_rep", "s_bar_div"]

    # Correlation heatmap
    corr = df[s_cols].corr()
    plt.figure(figsize=(6, 5))
    if has_seaborn:
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
    else:
        plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(s_cols)), s_cols, rotation=45, ha="right")
        plt.yticks(range(len(s_cols)), s_cols)
        plt.colorbar()
        for i in range(len(s_cols)):
            for j in range(len(s_cols)):
                plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center")
    plt.title("s-bar 4-dim correlation (across 10 domains)")
    plt.tight_layout()
    plt.savefig(output_dir / "sbar_correlation_matrix.png", dpi=150)
    plt.close()
    log.info(f"saved: {output_dir / 'sbar_correlation_matrix.png'}")

    # Bar chart per domain
    df_plot = df.melt(id_vars=["subset"], value_vars=s_cols, var_name="dim", value_name="value")
    plt.figure(figsize=(14, 6))
    if has_seaborn:
        sns.barplot(data=df_plot, x="subset", y="value", hue="dim")
    else:
        # fallback grouped bar
        subsets = df["subset"].tolist()
        x = np.arange(len(subsets))
        width = 0.2
        for i, col in enumerate(s_cols):
            plt.bar(x + i * width, df[col].values, width, label=col)
        plt.xticks(x + 1.5 * width, subsets, rotation=45, ha="right")
        plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.title("s-bar 4 dimensions across 10 Pile subsets")
    plt.tight_layout()
    plt.savefig(output_dir / "sbar_by_domain.png", dpi=150)
    plt.close()
    log.info(f"saved: {output_dir / 'sbar_by_domain.png'}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 1 · s-bar 4-component MVP")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--dataset_id", default="monology/pile-uncopyrighted")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Import heavy stuff here to speed up --help
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[env] torch={torch.__version__} device={device}")
    if device == "cuda":
        log.info(f"[env] GPU={torch.cuda.get_device_name(0)}")
    log.info(f"[model] loading {args.model_name}")
    model = SentenceTransformer(args.model_name, device=device)
    log.info(f"[model] ready")

    results = []
    for subset in args.subsets:
        log.info(f"========== {subset} ==========")
        try:
            r = process_subset(subset, args.n_samples, model, cache_dir)
            if r is not None:
                results.append(r)
        except Exception as e:
            log.error(f"[ERROR] subset '{subset}' failed: {e}")
            log.error(traceback.format_exc())

    if not results:
        log.error("no results, exiting")
        sys.exit(1)

    df = pd.DataFrame(results)
    csv_path = output_dir / "sbar_raw.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"saved: {csv_path}")
    log.info(f"\n=========== RESULTS ===========\n{df.to_string()}\n================================")

    try:
        make_plots(df, output_dir)
    except Exception as e:
        log.error(f"plot failed: {e}")
        log.error(traceback.format_exc())

    with open(output_dir / "sbar_details.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"saved: {output_dir / 'sbar_details.json'}")

    # Sanity checks
    log.info("========== Sanity checks ==========")
    for r in results:
        num_plus_rep = r.get("s_bar_num", 0) + r.get("s_bar_rep", 0)
        ok = abs(num_plus_rep - 1.0) < 1e-6 if not np.isnan(num_plus_rep) else False
        log.info(f"  {r['subset']}: s̄_num + s̄_rep = {num_plus_rep:.4f} ({'OK' if ok else 'FAIL'})")

    log.info("Stage 1 MVP complete.")


if __name__ == "__main__":
    main()
