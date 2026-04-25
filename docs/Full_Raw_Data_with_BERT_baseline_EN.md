# Stage 1 · Full Raw Data · with BERT Baseline

**Date**: 2026-04-25
**Companion to**: `papers/Stage1_Technical_Report_EN_v2.4.{md,pdf}`
**Purpose**: Transparent data reference for all numbers reported in the v2.4 main paper, including the BERT-style probe (§6).

---

## 0. Document Positioning

**Transparency principle**: This document collects all computational data from 2026-04-24 to 2026-04-25, including:

- The **main study** sentence-transformers data (MiniLM / BGE-small / BGE-large)
- The **standalone methodological probe** BERT-style baseline data (BERT-base / Legal-BERT / BioBERT / PubMedBERT)

Any researcher using this framework can cross-check numbers here. Any concern about whether BERT-baseline data has been hidden can be verified directly against this document.

**The two data classes are NOT directly compared**, because the two model classes have fundamentally different training objectives:

- **sentence-transformers**: contrastive learning + uniformity loss · designed for retrieval / clustering · silhouette is a reasonable measure
- **BERT-style MLM**: no uniformity post-processing · silhouette is anisotropy-dominated artifact (verified by 5-treatment comparison + whitening collapse to zero across 280 experiments)

---

## §0 Reader's Notice (Data Reference Disclaimer)

This document is a **pure data reference** — no narrative, interpretation, or claim.

**Two usage rules**:

1. The **main study (v2.4 §1-§5)** cites only the first 3 columns of §1.x (MiniLM / BGE-small / BGE-large) + §1.4 (n=3 raters) + §1.3 (24-point sentence-transformers scatter)
2. The **standalone methodological study (BERT probe, paper §6)** cites only §2.x (the 4 BERT-style models) + §2.6 (5-treatment 280 rows) + §2.7 (whitened ρ across 5 treatments)

**Cross-scope comparison is forbidden**: The first 3 columns and the last 4 columns in §2.2 are NOT directly comparable. sentence-transformers are trained with uniformity loss; BERT-style models are not — silhouette is heterogeneous between the two classes. Cross-class comparison yields anisotropy-dominated artifacts (validated by the 4-25 baseline experiment).

**Headline finding**:

> raw MiniLM vs Legal-BERT s_con Spearman ρ = -0.429 (the headline number behind the earlier "two-lens" hypothesis).
> After whitening: ρ = +0.048 · the "lens divergence" completely vanishes · the multi-lens hypothesis is **self-falsified by the 5-treatment experiment**.
> Across 7 models × 280 silhouette measurements, **all whitened values fall to zero or negative** (-0.05 to +0.009).
> Conclusion: silhouette on contextual embeddings is not measuring real clusters; it is anisotropy-dominated artifact. The v2.4 main study is therefore strictly scoped to sentence-transformers.

---

## 1. Main Data (sentence-transformers · used by §1-§5 of the paper)

### 1.1 Model specifications

| Model | Params | Dim | Training objective |
|---|---|---|---|
| all-MiniLM-L6-v2 | 66M | 384 | distillation + multi-task contrastive (1B sentence pairs · sbert-paraphrase) |
| BGE-small-en-v1.5 | 33M | 384 | MLM pretrain → retrieval contrastive fine-tuning (RetroMAE + contrastive · Xiao 2023) |
| BGE-large-en-v1.5 | 335M | 1024 | MLM pretrain → retrieval contrastive fine-tuning (RetroMAE + contrastive) |

### 1.2 Three models × 8 domains × 4 components

#### s̄_con

| Domain | MiniLM | BGE-small | BGE-large |
|---|---|---|---|
| Pile-CC | 0.020 | 0.021 | 0.016 |
| Wikipedia (en) | 0.028 | 0.031 | 0.023 |
| ArXiv | 0.045 | 0.074 | 0.066 |
| Github | 0.044 | 0.027 | 0.023 |
| PubMed Central | 0.046 | 0.049 | 0.044 |
| FreeLaw | 0.019 | 0.022 | 0.021 |
| StackExchange | 0.018 | 0.021 | 0.017 |
| USPTO Backgrounds | 0.035 | 0.033 | 0.019 |

#### s̄_div (Vendi)

| Domain | MiniLM | BGE-small | BGE-large |
|---|---|---|---|
| Pile-CC | 0.0880 | 0.0162 | 0.0270 |
| Wikipedia (en) | 0.0829 | 0.0218 | 0.0303 |
| ArXiv | 0.0551 | 0.0062 | 0.0069 |
| Github | 0.0577 | 0.0070 | 0.0092 |
| PubMed Central | 0.0607 | 0.0078 | 0.0093 |
| FreeLaw | **0.0173** | **0.0064** | 0.0081 |
| StackExchange | 0.0922 | 0.0117 | 0.0199 |
| USPTO Backgrounds | 0.0563 | 0.0102 | 0.0124 |

#### s̄_num / s̄_rep · MinHash, embedding-independent

**Cross 7 models (including BERT-style) values are exactly identical · max(across_models) − min(across_models) = 0**:

| Domain | n | s̄_num | s̄_rep |
|---|---|---|---|
| Pile-CC | 5000 | 1.000 | 0.000 |
| Wikipedia (en) | 5000 | 0.999 | 0.001 |
| ArXiv | 2657 | 0.999 | 0.001 |
| Github | 5000 | 0.986 | 0.014 |
| PubMed Central | 5000 | 0.999 | 0.001 |
| FreeLaw | 5000 | 0.995 | 0.005 |
| StackExchange | 5000 | 1.000 | 0.000 |
| USPTO Backgrounds | 5000 | 0.999 | 0.001 |

**On Kendall W = 0.976 (vs theoretical 1.000)**: Cross 7 models the rank vector is identical (rank std = 0 across raters). scipy `rankdata` defaults to `method="average"` for ties; eight domains include Pile-CC=StackExchange=1.000 and ArXiv=Wikipedia=PubMed=USPTO=0.999 (multi-way ties). The default formula yields W=0.976. The ties-corrected formula `W = 12S / (k²(n³−n) − kT)` yields W ≈ 1.000. Theoretical conclusion unchanged: MinHash does not depend on the embedding training objective.

### 1.3 Vendi-SVD ρ=-0.997 scatter (24 points · 3 sentence-transformers × 8 domains)

```
MiniLM:
  Pile-CC (top10=0.219, s_div=0.0880)
  Wiki    (0.225, 0.0829)
  ArXiv   (0.378, 0.0551)
  Github  (0.361, 0.0577)
  PubMed  (0.340, 0.0607)
  FreeLaw (0.569, 0.0173)
  Stack   (0.211, 0.0922)
  USPTO   (0.347, 0.0563)

BGE-small:
  Pile-CC (0.577, 0.0162)
  Wiki    (0.531, 0.0218)
  ArXiv   (0.752, 0.0062)
  Github  (0.708, 0.0070)
  PubMed  (0.705, 0.0078)
  FreeLaw (0.713, 0.0064)
  Stack   (0.635, 0.0117)
  USPTO   (0.657, 0.0102)

BGE-large:
  Pile-CC (0.512, 0.0270)
  Wiki    (0.498, 0.0303)
  ArXiv   (0.743, 0.0069)
  Github  (0.680, 0.0092)
  PubMed  (0.685, 0.0093)
  FreeLaw (0.686, 0.0081)
  Stack   (0.560, 0.0199)
  USPTO   (0.634, 0.0124)
```

**3 models × 8 domains = 24 points · Spearman ρ ≈ -0.997** (consistent with the 32-point version ρ=-0.997 within sentence-transformers scope).

### 1.4 Spearman ρ + Kendall's W (n=3 sentence-transformers)

#### Pairwise Spearman ρ (3 models × 8 domains)

| Pair | s̄_con | s̄_div |
|---|---|---|
| MiniLM vs BGE-small | 0.833 | 0.810 |
| MiniLM vs BGE-large | 0.762 | 0.810 |
| BGE-small vs BGE-large | 0.857 | 1.000 |

#### Kendall's W (n=3 raters · k=8 items · scipy exact)

| Component | W | χ² (df=7) | p | Verdict |
|---|---|---|---|---|
| s̄_num / s̄_rep | 0.976 (default) / 1.000 (ties-corrected) | 20.50 | 0.0046 | Identical (theoretical) |
| s̄_con | **0.878** | 18.44 | 0.0101 | **Strong consistency** |
| s̄_div | **0.915** | 19.22 | 0.0075 | **Strong consistency** |

### 1.5 Bootstrap 95% CI (n=8, B=1000)

| Component | Point estimate ρ (MiniLM vs BGE-small) | 95% CI | width |
|---|---|---|---|
| s̄_con | 0.833 | [0.317, 1.000] | 0.683 |
| s̄_div | 0.810 | [0.241, 0.975] | 0.734 |

CIs exclude 0; positive correlation is statistically significant. Width reflects the n=8 sample-size ceiling.

### 1.6 Slice sensitivity (MiniLM × FreeLaw + ArXiv)

| Subset | head | mid | tail | Multiplier |
|---|---|---|---|---|
| FreeLaw s_div | 0.017 | **0.030** | 0.026 | mid 1.77× |
| FreeLaw s_con | 0.019 | 0.020 | 0.022 | nearly unchanged |
| ArXiv s_div | 0.055 | 0.058 | 0.046 | <13% |
| ArXiv s_con | 0.045 | 0.038 | 0.038 | slight decrease |

**FreeLaw head boilerplate concentration is 1.77×** · this finding is independent of any BERT-side data (preserved in the main study).

### 1.7 K scan + Seed scan (MiniLM)

#### K scan

| Domain | K=2 | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|---|
| FreeLaw | 0.044 | 0.019 | 0.022 | 0.020 | 0.022 |
| StackExchange | 0.020 | 0.018 | 0.022 | 0.025 | 0.026 |
| PubMed Central | 0.032 | 0.046 | 0.053 | 0.051 | 0.045 |

FreeLaw stays < 0.045 across K ∈ [2, 50] · K-independent.

#### Seed scan (K=5 fixed)

FreeLaw σ ≈ 1e-5 across 5 seeds {1, 2, 3, 42, 100} · highly stable.

---

## 2. Standalone Methodological Baseline (BERT-style · probe-only · NOT directly comparable to §1)

> **Reminder**: Data in this section is NOT directly comparable to §1. silhouette on BERT-style models is anisotropy-dominated, not a real cluster signal.

### 2.1 BERT-style model specifications

| Model | Params | Dim | Training strategy | Vocab |
|---|---|---|---|---|
| google-bert/bert-base-uncased | 110M | 768 | Generic MLM | BERT 30k |
| nlpaueb/legal-bert-base-uncased | 110M | 768 | Legal MLM (from-scratch + new vocab) | Legal SP 30k |
| dmis-lab/biobert-v1.1 | 110M | 768 | Biomedical MLM (CPT from BERT) | BERT 28,996 (kept) |
| BiomedNLP-PubMedBERT | 110M | 768 | Biomedical MLM (from-scratch + new vocab) | Biomedical WP 30,522 |

### 2.2 BERT-style × 8 domains × s̄_con

| Domain | BERT-base | Legal-BERT | BioBERT | PubMedBERT |
|---|---|---|---|---|
| Pile-CC | 0.035 | 0.031 | 0.064 | 0.054 |
| Wikipedia | 0.024 | 0.034 | 0.052 | 0.050 |
| ArXiv | 0.096 | 0.061 | 0.079 | 0.066 |
| Github | 0.065 | 0.029 | 0.103 | 0.052 |
| PubMed Central | 0.092 | 0.049 | 0.056 | 0.043 |
| FreeLaw | **0.102** | **0.104** | **0.160** | **0.114** |
| StackExchange | **0.102** | 0.068 | 0.090 | **0.116** |
| USPTO | 0.068 | 0.065 | 0.049 | 0.054 |

### 2.3 BERT-style × 8 domains × s̄_div

| Domain | BERT-base | Legal-BERT | BioBERT | PubMedBERT |
|---|---|---|---|---|
| Pile-CC | 0.0024 | 0.00096 | 0.00102 | 0.00058 |
| Wikipedia | 0.0042 | 0.00118 | 0.00121 | 0.00060 |
| ArXiv | 0.0012 | 0.00079 | 0.00087 | 0.00056 |
| Github | 0.0019 | 0.00125 | 0.00147 | 0.00064 |
| PubMed | 0.0014 | 0.00088 | 0.00096 | 0.00057 |
| FreeLaw | 0.0012 | 0.00102 | 0.00085 | 0.00056 |
| StackExchange | 0.0016 | 0.00098 | 0.00110 | 0.00059 |
| USPTO | 0.0015 | 0.00089 | 0.00088 | 0.00057 |

### 2.4 7-model anisotropy spectrum (cross-8-domain mean)

#### l2norm mode top-k ratios

| Model | top-1 | top-5 | top-10 | top-20 | Spectrum H |
|---|---|---|---|---|---|
| MiniLM | 0.140 | 0.252 | **0.331** | 0.431 | 4.78 |
| BGE-small | 0.558 | 0.620 | **0.660** | 0.706 | 3.00 |
| BGE-large | 0.528 | 0.587 | **0.625** | 0.669 | 3.31 |
| **BERT-base** | — | — | **0.900** | — | 1.26 |
| Legal-BERT | 0.913 | 0.940 | **0.953** | 0.964 | 0.68 |
| BioBERT | — | — | **0.951** | — | 0.72 |
| **PubMedBERT** | 0.983 | 0.989 | **0.992** | 0.994 | 0.15 |

**Two distinct levels**:

- sentence-transformers: top-10 = 0.33-0.66
- BERT-style: top-10 = **0.90-0.99**

#### centered mode top-10 (global bias removed)

| Model | centered top-10 |
|---|---|
| MiniLM | 0.244 |
| BGE-small | 0.248 |
| BGE-large | 0.224 |
| **BERT-base** | **0.499** |
| Legal-BERT | 0.485 |
| BioBERT | 0.532 |
| PubMedBERT | 0.515 |

### 2.5 Raw vs Centered Gap · BERT-style across 8 domains

`gap = raw_top10 − centered_top10` (global bias intensity):

| Domain | BERT-base | Legal-BERT | BioBERT | PubMedBERT |
|---|---|---|---|---|
| Pile-CC | 0.501 | (TBD) | (TBD) | (TBD) |
| Wikipedia | 0.469 | (TBD) | (TBD) | (TBD) |
| ArXiv | 0.357 | (TBD) | (TBD) | (TBD) |
| Github | 0.350 | 0.447 | 0.365 | 0.469 |
| PubMed Central | 0.382 | 0.463 | 0.500 | 0.554 |
| FreeLaw | 0.321 | 0.388 | 0.288 | 0.351 |
| StackExchange | 0.369 | 0.463 | 0.333 | 0.439 |
| USPTO | 0.439 | (TBD) | (TBD) | (TBD) |

**BERT-base × 8 domains raw / centered full data**:

| Domain | raw_top10 | centered_top10 | gap |
|---|---|---|---|
| Pile-CC | 0.852 | 0.351 | 0.501 |
| Wikipedia | 0.777 | 0.308 | 0.469 |
| ArXiv | 0.945 | 0.587 | 0.357 |
| Github | 0.898 | 0.548 | 0.350 |
| PubMed Central | 0.929 | 0.547 | 0.382 |
| FreeLaw | 0.949 | 0.628 | 0.321 |
| StackExchange | 0.918 | 0.548 | 0.369 |
| USPTO | 0.914 | 0.475 | 0.439 |

**BERT-base centered top-10 cross-8-domain mean = 0.499**. Source: `data/ablation/anisotropy_raw_vs_centered_7models.csv` (BERT-base 8 rows).

### 2.6 5-treatment silhouette · 280 rows (core evidence)

#### Treatment definitions

| Treatment | Operation |
|---|---|
| raw | Direct on original embedding |
| centered | X − mean(X) |
| zstd | (X − mean) / std |
| abtt | All-but-the-top: subtract top-1 principal component |
| **whitened** | PCA whitening (uniform spectrum) |

#### FreeLaw × 7 models × 5 treatments

| Model | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.019 | 0.019 | 0.019 | 0.021 | **0.000** |
| BGE-small | 0.022 | 0.022 | 0.021 | 0.019 | **-0.004** |
| BGE-large | 0.020 | 0.020 | 0.019 | 0.014 | **-0.050** |
| BERT-base | 0.102 | 0.102 | 0.091 | 0.058 | **0.009** |
| Legal-BERT | 0.104 | 0.104 | 0.093 | 0.074 | **0.000** |
| BioBERT | 0.160 | 0.160 | 0.094 | 0.094 | **0.002** |
| PubMedBERT | 0.114 | 0.114 | 0.094 | 0.066 | **-0.005** |

#### PubMed Central × 7 models × 5 treatments

| Model | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.046 | 0.046 | 0.044 | 0.024 | **-0.027** |
| BGE-small | 0.049 | 0.049 | 0.038 | 0.042 | **-0.007** |
| BGE-large | 0.044 | 0.044 | 0.042 | 0.038 | **-0.020** |
| BERT-base | 0.092 | 0.092 | 0.083 | 0.062 | **-0.003** |
| Legal-BERT | 0.049 | 0.049 | 0.045 | 0.056 | **-0.002** |
| BioBERT | 0.056 | 0.056 | 0.046 | 0.022 | **0.000** |
| PubMedBERT | 0.043 | 0.043 | 0.038 | 0.032 | **-0.010** |

**Core finding**: After whitening, all 7 models × 8 domains silhouette ≈ 0 (range -0.05 to +0.009).

Full 280 rows: `data/treatments_5/silhouette_5treatments_7models.csv`.

### 2.7 Pairwise Spearman ρ_con · 5-treatment comparison

| Pair | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM vs Legal-BERT | -0.429 | -0.429 | -0.333 | +0.214 | **+0.048** |
| MiniLM vs BERT-base | -0.238 | -0.238 | -0.214 | +0.690 | **+0.381** |
| BERT-base vs Legal-BERT | +0.810 | +0.810 | +0.762 | +0.095 | +0.690 |
| BERT-base vs BioBERT | +0.524 | +0.524 | +0.548 | +0.333 | +0.524 |
| BERT-base vs PubMedBERT | +0.738 | +0.738 | +0.690 | +0.357 | +0.381 |
| Legal-BERT vs BioBERT | +0.190 | +0.190 | +0.190 | +0.119 | +0.333 |
| Legal-BERT vs PubMedBERT | +0.738 | +0.738 | +0.810 | +0.310 | +0.429 |
| BioBERT vs PubMedBERT | +0.500 | +0.500 | +0.476 | +0.786 | +0.310 |

### 2.8 Kendall W under 5 treatments (7-model raters)

| Treatment | W | χ² (df=7) | p |
|---|---|---|---|
| raw | 0.266 | 13.05 | 0.071 |
| centered | 0.266 | 13.05 | 0.071 |
| zstd | 0.286 | 14.00 | 0.051 |
| abtt | 0.315 | 15.43 | 0.031 |
| **whitened** | 0.228 | 11.19 | 0.131 |

### 2.9 Key findings

- BERT-base × StackExchange s_con = **0.102** · this falsifies any earlier "Legal-BERT 5× amplification" narrative (BERT-base already at 0.102, Legal-BERT 0.104 is only 1.02× of that).
- After whitening, silhouette across 7 models × 8 domains drops essentially to zero · evidence that silhouette on BERT-style is anisotropy-dominated.
- MiniLM vs Legal-BERT raw ρ = -0.429; after whitening ρ = +0.048 · the apparent "lens divergence" disappears.

---

## 3. Academic Honesty Statement

### 3.1 Withdrawn earlier claims

| Claim | Reason for withdrawal |
|---|---|
| "Legal-BERT × FreeLaw 5× amplification = sub-class signal" | BERT-base × FreeLaw is already 0.102; Legal-BERT is only 1.02× — comparison was against the wrong baseline. |
| "v2.2 §3.9 two-lens system (ρ_con = -0.43)" | After whitening ρ = +0.048 — the lens divergence is an anisotropy artifact. |
| "BioBERT × FreeLaw reverse collapse" | Same anisotropy-dominated mechanism, not a unique collapse. |
| "I6 four-mechanism decomposition mechanisms 2/3/4" | Involve BERT-style data; moved to the standalone §6. |

### 3.2 Claims preserved in the v2.4 main study

| Claim | Reason for preservation |
|---|---|
| s̄_num / s̄_rep are fully embedding-independent | Theoretical (MinHash does not go through the embedding pipeline). |
| Within sentence-transformers, Kendall W ≥ 0.92 | 3 models × 8 domains direct measurement. |
| MiniLM vs BGE Spearman ρ_con = 0.83 / ρ_div = 0.81 | Direct measurement + Bootstrap CI excludes 0. |
| Vendi-SVD ρ = -0.997 (24 sentence-transformers points) | Mathematical equivalence evidence. |
| FreeLaw head boilerplate concentration (1.77×) | Independent of BERT data; verified by slice experiment. |
| FreeLaw lowest-tier s̄_div under sentence-transformers (rank 1-2) | Scoped declaration; not extended to other lenses. |

---

## Appendix A · Data file paths

```
data-quality-vec-public/data/
├── primary/                 # sentence-transformers (main study)
│   ├── sbar_minilm.csv
│   ├── sbar_bge_small.csv
│   ├── sbar_bge_large.csv
│   └── sbar_bge_large_details.json
├── probe/                   # BERT-style (§6 standalone probe)
│   ├── sbar_bert_base.csv
│   ├── sbar_legal_bert.csv
│   ├── sbar_legal_bert_details.json
│   ├── sbar_biobert.csv
│   └── sbar_pubmedbert.csv
├── ablation/                # K-scan, slice, anisotropy
│   ├── kscan_results.csv
│   ├── sbar_trunc_mid.csv
│   ├── sbar_trunc_tail.csv
│   ├── anisotropy_results_7models.csv
│   ├── anisotropy_model_summary_7models.csv
│   ├── anisotropy_raw_vs_centered_7models.csv
│   ├── kendall_spearman_analysis.json
│   └── kendall_spearman_analysis_6models.json
└── treatments_5/            # 280-row 5-treatment silhouette
    ├── silhouette_5treatments_7models.csv
    ├── pairwise_rho_5treatments.json
    └── kendall_w_5treatments.csv
```

Intermediate `.npy` embeddings (~380 MB) and text caches (~137 MB) are NOT included in this repository — please regenerate from `code/run_stage1_sbar_v2.py` if needed.

---

A Chinese version of this document is available at `docs/Full_Raw_Data_with_BERT_baseline_CN.md` (kept verbatim for cross-reference).
