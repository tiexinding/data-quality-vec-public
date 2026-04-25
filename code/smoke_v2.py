"""
smoke_v2.py · 点火前 4 件验证 (B1 强烈建议)
============================================

S1 · Legal-BERT AutoModel + mean-pooling wrapper   [最易翻车]
S2 · BGE-large SentenceTransformer 加载             [1024 维]
S3 · v2 三种 truncation_pos 产出不同 embedding        [B1 #7 cache 污染防御]
S4 · 各向异性 SVD 可跑                                [明早 SVD 能跑]

每个 <30s · 4 件总 <3 min
"""
import sys, os, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/root/autodl-tmp")
from run_stage1_sbar_v2 import (
    HFMeanPoolEncoder, load_model, slice_text, compute_embeddings,
    compute_sbar_div,
)

CACHE = Path("/root/autodl-tmp/cache")
with open(CACHE / "texts_FreeLaw_5000.json") as f:
    TEXTS = json.load(f)
SAMPLE = TEXTS[:50]
print(f"[prep] loaded {len(SAMPLE)} FreeLaw texts for smoke")

# -------------- S1 : Legal-BERT wrapper --------------
print("\n===== S1 · Legal-BERT wrapper =====")
t0 = time.time()
try:
    enc = HFMeanPoolEncoder("nlpaueb/legal-bert-base-uncased", device="cuda")
    emb = enc.encode([slice_text(t, "head") for t in SAMPLE], batch_size=16, show_progress_bar=False)
    nan_count = int(np.isnan(emb).sum())
    print(f"  shape={emb.shape} dtype={emb.dtype} mean_norm={np.linalg.norm(emb,axis=1).mean():.3f}")
    print(f"  NaN count = {nan_count} (expect 0)")
    s_div = compute_sbar_div(emb, sub_n=50, seed=42)
    print(f"  vendi s_div on 50 = {s_div:.4f}")
    assert emb.shape == (50, 768), f"shape mismatch: {emb.shape}"
    assert nan_count == 0, "contains NaN"
    print(f"  [PASS] S1 in {time.time()-t0:.1f}s")
    del enc, emb
    import torch; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [FAIL] S1: {e}")
    raise

# -------------- S2 : BGE-large --------------
print("\n===== S2 · BGE-large load + 10-text encode =====")
t0 = time.time()
try:
    model, is_wrapper = load_model("BAAI/bge-large-en-v1.5", device="cuda")
    assert not is_wrapper, "BGE-large should not trigger wrapper"
    emb = model.encode([slice_text(t, "head") for t in SAMPLE[:10]],
                       batch_size=10, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=False)
    print(f"  shape={emb.shape} dtype={emb.dtype}")
    assert emb.shape == (10, 1024), f"shape mismatch: {emb.shape}"
    print(f"  [PASS] S2 in {time.time()-t0:.1f}s")
    del model, emb
    import torch; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [FAIL] S2: {e}")
    raise

# -------------- S3 : 三种 pos 产出不同 embedding --------------
print("\n===== S3 · head / mid / tail 产出差异 =====")
t0 = time.time()
try:
    # Use MiniLM (fast) to smoke-test pos
    model, _ = load_model("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    # 选长文本 (只 >= 3500 字符的, 保证 mid 有效)
    long_texts = [t for t in TEXTS if len(t) >= 3500][:20]
    print(f"  using {len(long_texts)} long texts (>=3500 chars)")
    emb_h = model.encode([slice_text(t, "head") for t in long_texts], show_progress_bar=False, normalize_embeddings=False)
    emb_m = model.encode([slice_text(t, "mid")  for t in long_texts], show_progress_bar=False, normalize_embeddings=False)
    emb_t = model.encode([slice_text(t, "tail") for t in long_texts], show_progress_bar=False, normalize_embeddings=False)
    # 计算 head/mid 和 head/tail 的 per-row cos sim
    def rowcos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (a * b).sum(axis=1)
    cos_hm = rowcos(emb_h, emb_m)
    cos_ht = rowcos(emb_h, emb_t)
    print(f"  mean cos(head, mid) = {cos_hm.mean():.3f}  (expect << 1.0)")
    print(f"  mean cos(head, tail)= {cos_ht.mean():.3f}  (expect << 1.0)")
    assert cos_hm.mean() < 0.95, "mid too similar to head — slice broken"
    assert cos_ht.mean() < 0.95, "tail too similar to head — slice broken"
    print(f"  [PASS] S3 in {time.time()-t0:.1f}s")
    del model
    import torch; torch.cuda.empty_cache()
except Exception as e:
    print(f"  [FAIL] S3: {e}")
    raise

# -------------- S4 : SVD 各向异性能算 --------------
print("\n===== S4 · SVD anisotropy on 50-vec dummy =====")
t0 = time.time()
try:
    from run_p0_1d_anisotropy import analyze
    # 用刚才 BGE-large 没存, 用 head embedding 临时造一个向量组试
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 128)).astype(np.float32)
    m = analyze(X)
    print(f"  isotropic random: top10_ratio={m['top10_ratio']:.3f} PR_norm={m['normalized_PR']:.3f}")
    # 各向异性: 前 3 维加强
    Y = X.copy(); Y[:, :3] *= 10
    m2 = analyze(Y)
    print(f"  anisotropic (first 3 dims ×10): top10_ratio={m2['top10_ratio']:.3f} PR_norm={m2['normalized_PR']:.3f}")
    assert m2['top10_ratio'] > m['top10_ratio'], "anisotropy detector broken"
    print(f"  [PASS] S4 in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  [FAIL] S4: {e}")
    raise

print("\n===== All 4 smoke tests PASSED · ready for launch =====")
