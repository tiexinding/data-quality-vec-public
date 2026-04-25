"""
阶梯 1 · s̄ 四分量 · v2 (ablation-ready)
=========================================

Adds over v1 (run_stage1_sbar.py):
  1. --use_wrapper / auto-detect Legal-BERT style models
     (AutoModel + mean-pooling instead of SentenceTransformer)
  2. --save_embeddings  → persists per-domain embeddings_{domain}.npy
     (5000 x D float32 per file) for downstream cross-embedding analysis
     (Vendi already subsamples to 2000, but we save all 5000)
  3. --truncation_pos {head,mid,tail} → changes text slice and cache filename
     - head: text[0:2000]        (matches v1 default)
     - mid : text[1500:3500]     (skips boilerplate prefix)
     - tail: text[-2000:]        (catches concluding boilerplate)
     Cache filename becomes texts_{subset}_{n}_{pos}.json to avoid B1 #7 pollution.

Result filename encoding (always under cache_dir):
  sbar_{subset}_{pos}.json      (sbar per subset · per truncation pos)
  embeddings_{subset}_{pos}.npy (if --save_embeddings)

B2 · 2026-04-24 · v2
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


DEFAULT_SUBSETS = [
    "Pile-CC", "Wikipedia (en)", "ArXiv", "Github",
    "PubMed Central", "FreeLaw", "StackExchange", "USPTO Backgrounds",
]


def _safe(name: str) -> str:
    return (name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", ""))


# ----------------------------------------------------------------------
# Text sampling with truncation-position-aware cache
# ----------------------------------------------------------------------
def load_and_sample(subset_name, n_samples, cache_dir, dataset_id="monology/pile-uncopyrighted",
                    max_scan=200000, log_every=10000, trunc_pos="head"):
    """For head/mid/tail positions, reuse raw texts cache (texts_{subset}_{n}.json)
    if present; positions only change how the *embedder* slices the text, not the
    stored text.  Returns raw texts (up to ~N chars each)."""
    from datasets import load_dataset
    from tqdm import tqdm

    # v1/v2 common raw-text cache (position-agnostic)
    cache_file = cache_dir / f"texts_{_safe(subset_name)}_{n_samples}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            texts = json.load(f)
        log.info(f"[cache] texts hit for {subset_name}: {len(texts)} samples")
        return texts

    log.info(f"[load] streaming {dataset_id}, filter='{subset_name}', n={n_samples}")
    ds = load_dataset(dataset_id, split="train", streaming=True)

    texts = []
    pbar = tqdm(total=n_samples, desc=f"sample {subset_name}")
    scanned = 0
    last_logged = 0
    for ex in ds:
        scanned += 1
        if scanned - last_logged >= log_every:
            log.info(f"[scan] {subset_name}: scanned={scanned}, matched={len(texts)}")
            last_logged = scanned
        if scanned >= max_scan:
            log.warning(f"[skip] {subset_name}: only {len(texts)} matched after {scanned} rows, giving up")
            break
        meta = ex.get("meta", {})
        if isinstance(meta, dict):
            label = meta.get("pile_set_name") or meta.get("subset") or meta.get("name")
        else:
            label = str(meta)
        if label == subset_name:
            text = (ex.get("text") or ex.get("content") or "").strip()
            if len(text) >= 50:
                texts.append(text)
                pbar.update(1)
                if len(texts) >= n_samples:
                    break
    pbar.close()

    log.info(f"[load] got {len(texts)} samples for {subset_name}")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)
    return texts


def slice_text(t, pos):
    """head / mid / tail slicing · each returns up to 2000 chars."""
    if pos == "head":
        return t[:2000]
    if pos == "mid":
        return t[1500:3500] if len(t) >= 1500 else t[:2000]
    if pos == "tail":
        return t[-2000:]
    raise ValueError(f"Unknown truncation_pos: {pos}")


# ----------------------------------------------------------------------
# Embeddings · SentenceTransformer or HF AutoModel + mean-pooling wrapper
# ----------------------------------------------------------------------
class HFMeanPoolEncoder:
    """Minimal wrapper for models without sentence-transformers pooling config
    (e.g. nlpaueb/legal-bert-base-uncased)."""
    def __init__(self, model_name, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # CVE-2025-32434 workaround: transformers/torch 2.5 拒 .bin; 本地预转 safetensors
        # use_safetensors=True + local_files_only=True 让它只用本地 safetensors
        try:
            self.model = AutoModel.from_pretrained(
                model_name, use_safetensors=True, local_files_only=True
            ).to(device).eval()
        except Exception:
            # fallback: 允许联网, 但强制 safetensors
            self.model = AutoModel.from_pretrained(
                model_name, use_safetensors=True
            ).to(device).eval()

    def encode(self, texts, batch_size=64, show_progress_bar=True,
               convert_to_numpy=True, normalize_embeddings=False, max_length=512):
        from tqdm import tqdm
        torch = self.torch
        all_embs = []
        it = range(0, len(texts), batch_size)
        if show_progress_bar:
            it = tqdm(list(it), desc="HF encode")
        for i in it:
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            if normalize_embeddings:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)


def _is_wrapper_model(model_name):
    mn = model_name.lower()
    # Known non-sentence-transformers models that need our wrapper
    keywords = ["legal-bert", "scibert", "biobert"]
    return any(k in mn for k in keywords)


def load_model(model_name, device, force_wrapper=False):
    if force_wrapper or _is_wrapper_model(model_name):
        log.info(f"[model] using HF AutoModel + mean-pooling wrapper for {model_name}")
        return HFMeanPoolEncoder(model_name, device=device), True
    else:
        from sentence_transformers import SentenceTransformer
        log.info(f"[model] using SentenceTransformer for {model_name}")
        return SentenceTransformer(model_name, device=device), False


def compute_embeddings(texts, model, batch_size=64, trunc_pos="head"):
    texts_trunc = [slice_text(t, trunc_pos) for t in texts]
    log.info(f"[embed] encoding {len(texts_trunc)} texts (pos={trunc_pos}), batch_size={batch_size}")
    emb = model.encode(
        texts_trunc,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float32)


# ----------------------------------------------------------------------
# s̄ components (unchanged from v1 · documented hardcodes stay)
# ----------------------------------------------------------------------
def compute_sbar_con(embeddings, k=5, sample_size=5000, random_state=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = len(embeddings)
    if n < k + 1:
        return float("nan")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    sub = min(sample_size, n)
    return float(silhouette_score(embeddings, labels, sample_size=sub, random_state=random_state))


def compute_sbar_num(texts, threshold=0.8, num_perm=128):
    from datasketch import MinHash, MinHashLSH
    from tqdm import tqdm
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_count = 0
    for i, text in enumerate(tqdm(texts, desc="MinHash LSH")):
        m = MinHash(num_perm=num_perm)
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


def compute_sbar_div(embeddings, sub_n=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    if n == 0:
        return float("nan")
    if n > sub_n:
        idx = rng.choice(n, sub_n, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    emb = emb / norm
    K = emb @ emb.T
    K = K / len(emb)
    eig = np.linalg.eigvalsh(K)
    eig = np.clip(eig, 0.0, None)
    s = eig.sum()
    if s <= 0:
        return float("nan")
    p = eig / s
    H = -np.sum(p * np.log(p + 1e-12))
    vendi = float(np.exp(H))
    return vendi / len(emb)


# ----------------------------------------------------------------------
# Per-subset pipeline
# ----------------------------------------------------------------------
def process_subset(subset_name, n_samples, model, cache_dir, trunc_pos="head",
                   save_embeddings=False, emb_dir=None):
    safe = _safe(subset_name)
    pos_tag = trunc_pos
    result_file = cache_dir / f"sbar_{safe}_{pos_tag}.json"
    emb_file = (emb_dir / f"embeddings_{safe}_{pos_tag}.npy") if (save_embeddings and emb_dir) else None

    if result_file.exists() and (not save_embeddings or (emb_file and emb_file.exists())):
        with open(result_file) as f:
            r = json.load(f)
        log.info(f"[cache] result hit for {subset_name}@{pos_tag}: {r}")
        return r

    texts = load_and_sample(subset_name, n_samples, cache_dir, trunc_pos=trunc_pos)
    if len(texts) < 50:
        log.warning(f"[skip] {subset_name}: only {len(texts)} samples")
        return None

    embeddings = compute_embeddings(texts, model, trunc_pos=trunc_pos)

    if save_embeddings and emb_file is not None:
        emb_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_file, embeddings)
        log.info(f"[save] {emb_file} shape={embeddings.shape} dtype={embeddings.dtype}")

    r = {
        "subset": subset_name,
        "n": len(texts),
        "trunc_pos": pos_tag,
        "s_bar_con": compute_sbar_con(embeddings),
        "s_bar_num": compute_sbar_num(texts),
        "s_bar_div": compute_sbar_div(embeddings),
    }
    r["s_bar_rep"] = 1.0 - r["s_bar_num"] if not np.isnan(r["s_bar_num"]) else float("nan")

    with open(result_file, "w") as f:
        json.dump(r, f, indent=2)
    log.info(f"[done] {subset_name}@{pos_tag} -> {r}")
    return r


# ----------------------------------------------------------------------
# Plotting (same as v1 but adapted to include pos in title)
# ----------------------------------------------------------------------
def make_plots(df, output_dir, model_name="?", trunc_pos="head"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    s_cols = ["s_bar_con", "s_bar_num", "s_bar_rep", "s_bar_div"]
    corr = df[s_cols].corr()
    plt.figure(figsize=(6, 5))
    if has_seaborn:
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
    else:
        plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.xticks(range(len(s_cols)), s_cols, rotation=45, ha="right")
        plt.yticks(range(len(s_cols)), s_cols)
        plt.colorbar()
    plt.title(f"s-bar corr · {Path(model_name).name} · {trunc_pos}")
    plt.tight_layout()
    plt.savefig(output_dir / "sbar_correlation_matrix.png", dpi=150)
    plt.close()

    df_plot = df.melt(id_vars=["subset"], value_vars=s_cols, var_name="dim", value_name="value")
    plt.figure(figsize=(14, 6))
    if has_seaborn:
        sns.barplot(data=df_plot, x="subset", y="value", hue="dim")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"s-bar × domain · {Path(model_name).name} · {trunc_pos}")
    plt.tight_layout()
    plt.savefig(output_dir / "sbar_by_domain.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 1 v2 · ablation-ready s-bar runner")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--emb_dir", default=None,
                        help="If --save_embeddings, write .npy files here (default: output_dir)")
    parser.add_argument("--dataset_id", default="monology/pile-uncopyrighted")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save per-domain embedding matrices as .npy")
    parser.add_argument("--truncation_pos", default="head", choices=["head", "mid", "tail"])
    parser.add_argument("--use_wrapper", action="store_true",
                        help="Force HF AutoModel + mean-pooling wrapper (auto-on for legal-bert*)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    emb_dir = Path(args.emb_dir) if args.emb_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.save_embeddings:
        emb_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"[env] torch={torch.__version__} device={device}")
    if device == "cuda":
        log.info(f"[env] GPU={torch.cuda.get_device_name(0)}")

    log.info(f"[cfg] model={args.model_name} trunc_pos={args.truncation_pos} "
             f"save_emb={args.save_embeddings} use_wrapper={args.use_wrapper}")

    model, is_wrapper = load_model(args.model_name, device, force_wrapper=args.use_wrapper)
    log.info(f"[model] ready (wrapper={is_wrapper})")

    results = []
    for subset in args.subsets:
        log.info(f"========== {subset} ==========")
        try:
            r = process_subset(subset, args.n_samples, model, cache_dir,
                               trunc_pos=args.truncation_pos,
                               save_embeddings=args.save_embeddings,
                               emb_dir=emb_dir)
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
        make_plots(df, output_dir, model_name=args.model_name, trunc_pos=args.truncation_pos)
    except Exception as e:
        log.error(f"plot failed: {e}\n{traceback.format_exc()}")

    with open(output_dir / "sbar_details.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("========== Sanity checks ==========")
    for r in results:
        nr = r.get("s_bar_num", 0) + r.get("s_bar_rep", 0)
        ok = abs(nr - 1.0) < 1e-6 if not np.isnan(nr) else False
        log.info(f"  {r['subset']}: s̄_num + s̄_rep = {nr:.4f} ({'OK' if ok else 'FAIL'})")

    log.info("Stage 1 v2 complete.")


if __name__ == "__main__":
    main()
