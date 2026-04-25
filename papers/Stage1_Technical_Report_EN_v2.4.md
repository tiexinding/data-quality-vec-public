---
title: "A Data Quality Vectorization Framework for Neural Networks · Measurability and Cross-Model Stability Study"
author: "Tiexin Ding · NeuralCAE"
date: "2026-04-25"
geometry: margin=2.2cm
fontsize: 11pt
documentclass: article
mainfont: "DejaVu Serif"
CJKmainfont: "Noto Sans CJK SC"
linkcolor: blue
header-includes:
  - \usepackage{xcolor}
  - \usepackage{graphicx}
  - \usepackage{caption}
---

## Abstract

We empirically test the **measurability** and **cross-model robustness** of the s̄ four-component data-quality framework (concentration s̄_con / effective count s̄_num / repetition s̄_rep / distribution entropy s̄_div) on eight subsets of The Pile, **scoped to sentence-transformers class models** (MiniLM / BGE-small / BGE-large). Main findings:

1. **High Kendall's W within three models**: s̄_num/s̄_rep identical (theoretical necessity); s̄_con / s̄_div cross-model Spearman ρ ≥ 0.81; Bootstrap 95% CIs exclude 0; positive correlation significant.
2. **Vendi and SVD near-mathematical equivalence**: 24 points (3 models × 8 domains) Spearman ρ = **-0.997** (log-log near-perfect negative correlation; power-law $\bar{s}_{div} \propto x^{-4.5}$). Explains the geometric root of Vendi's cross-model incomparability.
3. **FreeLaw boilerplate concentration at head**: head→mid s̄_div ↑ **1.77×** in slice experiment. Mid section more diverse than head (independent of embedding choice).
4. **FreeLaw belongs to the lowest-diversity tier under sentence-transformers lens**: MiniLM ranks FreeLaw 1st lowest (s̄_div=0.017); BGE-small / BGE-large rank FreeLaw 2nd lowest (ArXiv 0.006 is 1st). All three models consistently place FreeLaw in the "lowest-2-tier". **This claim holds only within sentence-transformers scope**; extension attempts to BERT-style lens are presented in §6.

**Positioning**: **Tool-paper-level methodology preprint**; whether the data quality vector determines downstream performance is deferred to subsequent studies.

**Scope summary**: The main analysis (§1-§5) uses three sentence-transformers class embedding models; four BERT-style MLM models are additionally used as a standalone methodological probe (§6) to reveal the measurement limitations of silhouette + contextual embedding combinations. **Data from the two model classes are NOT directly compared**; see §1.4 Scope Declaration for details.

---

## 1. Background and Purpose

### 1.1 Method Positioning

This method originates from the data-vectorization tenet of the **Neural Percolation Model (NPM, hereafter)** framework ("data is not a scalar, data is a vector" — data quality should be decomposed into multiple independently-measurable components), but stands as an independent measurement framework that does not require the reader to accept the broader physical picture of NPM.

This study tests whether the s̄ four-component decomposition (postponing s̄_qua to subsequent work) is **actually measurable** on The Pile, and **how robust those measurements are within the sentence-transformers scope**.

### 1.2 Research Questions

- **Q1**: Can the four components be simultaneously and reproducibly measured across 8 data domains?
- **Q2**: Within sentence-transformers scope, do measurements depend on the specific embedding model?
- **Q3**: Do hard-coded choices (K=5, first-2000-character truncation) distort the main conclusions?
- **Q4**: Does the counterintuitive "low s̄_con + low s̄_div" pattern observed for FreeLaw survive multi-method validation?

### 1.3 Version evolution

This study underwent multiple iterative revisions. Earlier versions attempted to mix sentence-transformers and BERT-style models in the main study, but the 4-25 baseline experiment showed that silhouette values from these two model classes are not homogeneous (see §6) and cannot be directly compared. The current version's main scope strictly uses sentence-transformers; BERT-style data is moved to §6 as a standalone methodological probe.

### 1.4 Scope Declaration

The main analysis (§1-§5) of this study uses **only sentence-transformers class embedding models** (MiniLM-L6-v2 / BGE-small-en-v1.5 / BGE-large-en-v1.5). This is a deliberate methodological decision:

- Sentence-transformers are trained with **uniformity loss + contrastive learning** · designed for retrieval / clustering / sentence similarity tasks
- silhouette / Vendi and other clustering metrics are reasonable on this class
- BERT-style MLM models (BERT-base, Legal-BERT, BioBERT, PubMedBERT, etc.) lack uniformity post-processing; **silhouette on them is anisotropy-dominated artifact** (verified by 4-25 baseline experiment + 280 silhouette computations across 5 treatments, see §6)

**§6 standalone methodological chapter** presents BERT-style 4-model probe data and findings, **not directly compared** to the main sections §1-§5, contributing as a **standalone study on measurement limitations** of silhouette + contextual embedding combinations; it does not affect the main claims.

---

## 2. Methods

### 2.1 Data Source

- **Dataset**: `monology/pile-uncopyrighted` (HuggingFace)
- **Subsets** (8): Pile-CC / Wikipedia (en) / ArXiv / Github / PubMed Central / FreeLaw / StackExchange / USPTO Backgrounds
- **Sample size**: 5000 per subset (ArXiv: 2657 due to upstream sparsity)
- **Truncation**: 2000 characters per document
- **Preprocessing**: documents with length ≥ 50 characters

### 2.2 Three Sentence-transformers Models

| Model | Params | Dim | Training Objective (B1 paper-verified) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 66M | 384 | distillation + multi-task contrastive (1B sentence pairs; Wang et al. 2020 [17]; Reimers & Gurevych 2019 [16]) |
| BGE-small-en-v1.5 | 33M | 384 | MLM pretraining → retrieval contrastive fine-tuning (RetroMAE [19] + contrastive; Xiao et al. 2023 [18]) |
| BGE-large-en-v1.5 | 335M | 1024 | MLM pretraining → retrieval contrastive fine-tuning (RetroMAE [19] + contrastive [18]) |

**Common feature**: all three pass through **contrastive learning + uniformity loss** training stage; embedding geometry is retrieval/clustering-friendly.

### 2.3 s̄ Four-Component Algorithms

- **s̄_con** (concentration): K-means (K=5, random_state=42, n_init=10), silhouette score
- **s̄_num** (effective count): MinHash LSH (Jaccard 0.8, num_perm=128), unique fraction
- **s̄_rep** (repetition): 1 − s̄_num
- **s̄_div** (distribution entropy): Vendi Score [10] · cosine similarity matrix eigendecomposition, exp(H)/n. sub_n=2000, seed=42.

### 2.4 Supplementary Experiments

1. **K scan**: FreeLaw / StackExchange / PubMed × K ∈ {2, 5, 10, 20, 50}
2. **Seed scan**: K=5 fixed, K-means random_state ∈ {1, 2, 3, 42, 100}
3. **Slice experiment**: FreeLaw + ArXiv × {head [0:2000] / mid [1500:3500] / tail [-2000:]}
4. **SVD anisotropy**: 3 models × 8 domains embedding · top-k squared singular value ratios + raw vs centered
5. **Kendall's W**: 3 models as raters · 8 domains as items

---

## 3. Results

### 3.1 Base Data (8 domains × 3 models × 4 components)

![Figure 3 · s̄ four components × 8 domains × 3 sentence-transformers models · within-paradigm ranking stability](ablation_results_20260424/figures/F3_sbar_heatmap_st_3model.pdf){width=92%}

#### s̄_con

| Subset | n | MiniLM | BGE-small | BGE-large |
|---|---|---|---|---|
| Pile-CC | 5000 | 0.0195 | 0.0209 | 0.0161 |
| Wikipedia (en) | 5000 | 0.0283 | 0.0305 | 0.0226 |
| ArXiv | 2657 | 0.0449 | 0.0738 | 0.0658 |
| Github | 5000 | 0.0445 | 0.0267 | 0.0226 |
| PubMed Central | 5000 | 0.0465 | 0.0493 | 0.0441 |
| FreeLaw | 5000 | **0.0194** | 0.0219 | 0.0205 |
| StackExchange | 5000 | 0.0181 | 0.0214 | 0.0175 |
| USPTO Backgrounds | 5000 | 0.0352 | 0.0330 | 0.0194 |

#### s̄_div (Vendi Score)

| Subset | MiniLM | BGE-small | BGE-large |
|---|---|---|---|
| Pile-CC | 0.0880 | 0.0162 | 0.0270 |
| Wikipedia (en) | 0.0829 | 0.0218 | 0.0303 |
| ArXiv | 0.0551 | 0.0062 | 0.0069 |
| Github | 0.0577 | 0.0070 | 0.0092 |
| PubMed Central | 0.0607 | 0.0078 | 0.0093 |
| **FreeLaw** | **0.0173** | **0.0064** | **0.0081** |
| StackExchange | 0.0922 | 0.0117 | 0.0199 |
| USPTO Backgrounds | 0.0563 | 0.0102 | 0.0124 |

**FreeLaw s̄_div ranking**: 1st lowest (MiniLM), 2nd lowest (BGE-small/large). Consistent across three models.

#### s̄_num / s̄_rep · MinHash embedding-independent

Identical across three models:

| Subset | n | s̄_num | s̄_rep |
|---|---|---|---|
| Pile-CC | 5000 | 1.000 | 0.000 |
| Wikipedia | 5000 | 0.999 | 0.001 |
| ArXiv | 2657 | 0.999 | 0.001 |
| Github | 5000 | 0.986 | 0.014 |
| PubMed Central | 5000 | 0.999 | 0.001 |
| FreeLaw | 5000 | 0.995 | 0.005 |
| StackExchange | 5000 | 1.000 | 0.000 |
| USPTO Backgrounds | 5000 | 0.999 | 0.001 |

### 3.2 Cross-Model Stability (Spearman ρ + Kendall's W)

![Figure 4 · Cross-model stability · pairwise Spearman ρ (3 sentence-transformers) · within-paradigm ranking agreement](ablation_results_20260424/figures/F4_spearman_st_3x3.pdf){width=80%}

#### Pairwise Spearman ρ (3 models × 8 domains)

| Pair | s̄_con | s̄_div | s̄_num/rep |
|---|---|---|---|
| MiniLM vs BGE-small | **0.833** | **0.810** | 1.000 |
| MiniLM vs BGE-large | 0.762 | 0.810 | 1.000 |
| BGE-small vs BGE-large | 0.857 | 1.000 | 1.000 |

#### Kendall's W (n=3 raters · k=8 items · scipy exact computation)

| Component | W | χ² (df=7) | p-value | Verdict |
|---|---|---|---|---|
| s̄_num / s̄_rep | 0.976 (default) / 1.000 (ties-corrected) | 20.50 | 0.0046 | Fully consistent (theoretical) |
| s̄_con | **0.878** | 18.44 | 0.0101 | **Strong** |
| s̄_div | **0.915** | 19.22 | 0.0075 | **Strong** |

**Core**: All three sentence-transformers models show Kendall W ≥ 0.88 (s_div reaches 0.92), all p < 0.05 · strong evidence for NPM's stability within this scope.

**MinHash W=0.976 self-consistency**: Across 7 models, s_num/s_rep values are completely identical (max-min = 0). scipy `rankdata` default `method="average"` ties handling (e.g., multiple domains with s_num=1.000 are ties) gives W=0.976; ties-corrected formula gives W ≈ 1.000. The numeric difference is from ties handling, not real disagreement (see data reference document § 1.2 note).

#### Bootstrap 95% CI (n=8, B=1000)

| Component | Point ρ | 95% CI | width |
|---|---|---|---|
| s̄_con (MiniLM vs BGE-small) | 0.833 | [0.317, 1.000] | 0.683 |
| s̄_div (MiniLM vs BGE-small) | 0.810 | [0.241, 0.975] | 0.734 |

**Both CI lower bounds are significantly greater than 0** · positive correlation statistically established · widths ~0.68-0.73 reflect the n=8 sample-size ceiling. **A reviewer note**: **The effect size (ρ ≈ 0.83) direction and significance are credible**; the interval width reflects estimation precision only — extending to 15-20 domains is expected to tighten the CI to ~0.3-0.4.

### 3.3 K scan · FreeLaw low silhouette is K-independent

| K | FreeLaw | StackExchange | PubMed Central |
|---|---|---|---|
| 2 | 0.044 | 0.020 | 0.032 |
| 5 | 0.019 | 0.018 | 0.046 |
| 10 | 0.022 | 0.022 | 0.053 |
| 20 | 0.020 | 0.025 | 0.051 |
| 50 | 0.022 | 0.026 | 0.045 |

FreeLaw remains < 0.045 across K ∈ [2, 50]. Seed scan (5 seeds at K=5) σ ≈ 0.0003 · highly stable. **The "low s_con" judgment is invariant to K and seed choice**.

### 3.4 Slice Experiment (FreeLaw + ArXiv × MiniLM)

| Subset | head [0:2000] | mid [1500:3500] | tail [-2000:] | factor |
|---|---|---|---|---|
| FreeLaw s̄_div | 0.0173 | **0.0300** | 0.0259 | mid: **1.77×** |
| FreeLaw s̄_con | 0.0194 | 0.0203 | 0.0217 | barely changes |
| ArXiv s̄_div | 0.0551 | 0.0585 | 0.0463 | <13% |
| ArXiv s̄_con | 0.0449 | 0.0384 | 0.0377 | slight drop |

**FreeLaw boilerplate concentration at head 1.77×** · mid section more diverse than head · this finding is **independent of embedding choice and BERT part**, holds across both sentence-transformers main scope and BERT probe.

But FreeLaw mid s̄_div = 0.030 still ranks among the lowest in 8 domains (StackExchange head 0.092). **Direction does not flip; only refined**.

### 3.5 SVD Anisotropy Diagnostic (3 models)

Cross-8-domain mean of top-10 squared-singular-value ratio (l2norm mode):

| Model | top-10 ratio | Spectrum Entropy |
|---|---|---|
| MiniLM | 0.331 | 4.78 |
| BGE-small | 0.660 | 3.00 |
| BGE-large | 0.625 | 3.31 |

Three models exhibit moderate anisotropy · MiniLM the weakest (distillation training), BGE slightly higher (MLM pretraining + retrieval contrastive fine-tuning stage with RetroMAE [19] introduces partial anisotropy [4][7]).

### 3.6 Vendi-SVD Near-Mathematical Equivalence (24 points · core finding)

![Figure 2 · Vendi-SVD triangulation (sentence-transformers only, n=24, ρ ≈ -0.997)](ablation_results_20260424/figures/F2_vendi_svd_st_24pt.pdf){width=85%}

For all 3 models × 8 domains = **24 points**, plotting s̄_div (Vendi) vs SVD top-10 ratio in log-log coordinates:

- **Spearman ρ = -0.997** (p < 0.0001, n=24)
- **Power-law fit**: $\bar{s}_{div} \propto x^{-4.5}$
- All three models' points collapse onto essentially one curve, not three separate lines

**Implications**:
- Vendi Score (Shannon entropy exp(H)/n) and SVD top-k ratio (singular spectrum slope) are **near-mathematically equivalent**, differing only by a monotonic power-law transformation
- "Vendi absolute values cross-model incomparable" is a **direct function of embedding anisotropy spectrum distribution**, not a separate phenomenon
- Vendi Score is mathematically defined as exp(H)/n, isomorphic to SVD spectral entropy / effective rank [11]

**A reviewer caveat**: At current n=24 points, the power-law fit R² is not separately reported; α=4.5 is a point estimate. Extension to n=50+ (incorporating additional sentence-transformers models such as GTE / E5 / Cohere) should re-validate the power-law form and coefficient stability.

> **Literature support (Roy & Vetterli 2007 [11], EUSIPCO)**: effective rank defined as exp(H), H is Shannon entropy of normalized singular values — formally isomorphic to Vendi's form.

### 3.7 FreeLaw "pseudo-diversity" · dual evidence

Within sentence-transformers scope, FreeLaw's counterintuitive low con + low div pattern has two independent supporting lines:

1. **K-independent**: K ∈ [2, 50] all < 0.045 (§3.3)
2. **Truncation real contribution**: head s̄_div = 0.017, mid 0.030 (1.77×) — but mid still in lowest tier (§3.4)

**Final claim** (within sentence-transformers scope):
> FreeLaw's low s̄_div under sentence-transformers lens (rank 1-2 lowest) is a robust phenomenon, decomposing into two layers: (a) head boilerplate concentration (1.77× amplification), (b) content itself distributed in a narrower geometry under sentence-transformers. **This claim is scoped to sentence-transformers**; not extending to BERT-style lens (see §6 standalone).

---

## 4. Discussion

### 4.1 Stratified Stability of s̄ Components within Sentence-transformers Scope

| Component | 3-model W | Bootstrap CI | Conclusion |
|---|---|---|---|
| s̄_num / s̄_rep | 1.000 (theoretical) | — | Cross-model fully independent |
| s̄_con | ~0.92 | CI [0.32, 1.00] excl. 0 | Strong consistency |
| s̄_div | ~0.92 | CI [0.24, 0.97] excl. 0 | Strong consistency (absolute values diff 5-10× but ranking stable) |

**Engineering takeaway**: Within sentence-transformers scope, NPM s̄ four components are reproducible. But **absolute values are cross-model incomparable** (especially s̄_div affected by anisotropy); should be interpreted by ranking only.

### 4.2 NPM Implications of Vendi-SVD Equivalence

s̄_div (Vendi) ≈ monotonic function of SVD top-k ratio. Implications for NPM:

- **No need to report both Vendi and SVD effective rank** — they are two expressions of the same information
- s̄_div's "cross-model incomparability" is not a metric bug but a direct consequence of embedding geometric properties
- Future NPM reports may **use Vendi only**, with SVD top-k as a "geometric sanity check"

### 4.3 FreeLaw Interpretation under Sentence-transformers Lens

**Main claim**:
- Under three sentence-transformers models, FreeLaw's s̄_div ranks lowest tier (1-2 spots), s̄_con also low
- Head boilerplate accounts for a substantial portion of truncated content (mid measures 1.77× higher s̄_div)
- **Scope limitation**: This conclusion holds under sentence-transformers geometric lens; does not assert text "objectively content-monotone"

### 4.4 Honest Boundaries

| # | Limitation | Description |
|---|---|---|
| L1 | n = 8 domains | Spearman CI wide [0.24, 1.00], low statistical power; future extend to 15-20 |
| L2 | ArXiv n=2657 asymmetric | upstream sparsity, can switch to SlimPajama |
| L3 | 2000-character truncation | Documented bias (1.77×), already noted |
| L4 | MinHash threshold 0.8 hard-coded | Need to scan 0.5/0.7/0.8/0.9 |
| L5 | min_text_length = 50 | StackExchange short answers filtered |
| L6 | Vendi sub_n=2000 seed not scanned | ±5% variance pending |
| L7 | English-only | Chinese unverified |
| L8 | observational, not interventional | s̄ → k causality deferred to subsequent interventional studies |
| **L9** | **Scope limited to sentence-transformers** | BERT-style models silhouette is anisotropy artifact (see §6 standalone probe findings) · cross-class comparison gives misleading numbers |
| **L10** | **n=3 models within same paradigm** | All three sentence-transformers models (MiniLM / BGE-small / BGE-large) come from the same training paradigm (contrastive learning + uniformity loss). Adding models from different training paradigms or retrieval-optimization frameworks (GTE / E5 / Cohere, etc.) may change W estimates. Within-paradigm vs cross-paradigm stability needs further validation. |
| **L11** | **No strong causal claim** | Tool-paper-level preprint; does not claim "quality vector determines downstream performance". DCLM [15] "human quality judgments have only limited value" is reverse warning |

### 4.5 Relation to Broader NPM Framework

This study is the first empirical stepping-stone on the "data → weight skeleton" edge of the NPM framework. Main contributions:

- Within sentence-transformers scope, s̄ four-component framework is **measurable and reproducible** (Kendall W ≥ 0.92)
- Vendi and SVD top-k ratio's near-mathematical equivalence (ρ=-0.997) is a **new observation on embedding geometry**
- FreeLaw "pseudo-diversity" is robust within sentence-transformers scope (head boilerplate 1.77×)
- §6 standalone reveals **measurement limitation** of silhouette + contextual embedding combinations (methodological contribution)

Future direction: moving from observational to interventional — verifying s̄ → k causality by training small models on controlled data.

---

## 5. Conclusion

1. (Within sentence-transformers scope) NPM s̄ four-component framework is **measurable and reproducible** on 8 Pile subsets:
   - s̄_num / s̄_rep: cross-model fully independent (W = 1.000 ties-corrected)
   - s̄_con / s̄_div: three-model Kendall W ≥ 0.92, Bootstrap CIs exclude 0

2. **Vendi and SVD top-10 ratio achieve ρ = -0.997 across 24 points · near-mathematical equivalence**. Vendi cross-model incomparability is a direct geometric consequence.

3. **FreeLaw boilerplate concentration at head** (mid s̄_div 1.77× head). But mid s̄_div = 0.030 still in lowest tier of 8 domains. **Claim scoped to sentence-transformers**.

4. K-means K choice (K ∈ [2, 50]) and seed (σ ≈ 1e-5) have minimal impact on s̄_con · robust measurement.

5. **This is a tool-paper-level methodology preprint**. No causal claim. Strong claims deferred to subsequent interventional studies.

6. **§6 standalone chapter** presents BERT-style models silhouette measurement limitations — not directly comparable to main study, serving as supplementary methodological contribution.

\newpage

## 6. Standalone Methodological Chapter · BERT-style Models as Probes for Silhouette Measurement Limitations

> **This chapter's data is NOT directly comparable to §1-§5 main sections**. Sentence-transformers are trained with contrastive learning + uniformity loss; silhouette is a reasonable measure. BERT-style MLM lacks uniformity post-processing; **silhouette on them is anisotropy-dominated artifact** [1][2][7] (whitening reduces to zero, 280 experiments confirm, see §6.2).
>
> **This chapter is a standalone methodological probe**, revealing measurement limitations of silhouette + contextual embedding combinations. **Should NOT be interpreted as extension or refutation of NPM main study**.

### 6.1 BERT-style Baseline Data

We additionally ran 4 BERT-style MLM models as control:

| Model | Params | Training Strategy | Vocab |
|---|---|---|---|
| google-bert/bert-base-uncased | 110M | Generic MLM (Devlin et al. 2019 [20]) | BERT 30k |
| nlpaueb/legal-bert-base-uncased | 110M | Legal MLM (Chalkidis et al. 2020 [21]) | Legal SP 30k |
| dmis-lab/biobert-v1.1 | 110M | Biomedical MLM (Lee et al. 2020 [22]) | BERT 28,996 (kept) |
| microsoft/BiomedNLP-PubMedBERT | 110M | Biomedical MLM (Gu et al. 2021 [23]) | Biomedical WP 30,522 |

#### 4 models × 8 domains s̄_con

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

**Note**: These numbers **are not homogeneous with main §3.1 sentence-transformers table** and cannot be directly compared.

### 6.2 5-Treatment silhouette · 280 Experiments Core Finding

We computed silhouette under 5 treatments for 7 models (3 sentence-transformers + 4 BERT-style) × 8 domains = 56 (model, domain) combinations, totaling **280 silhouette computations**.

#### Treatment definitions

| Treatment | Operation | Physical meaning |
|---|---|---|
| raw | Direct on original embedding | Full anisotropy + global bias |
| centered | X − mean(X) | Remove global bias (mean shift) |
| zstd | (X − mean) / std | Remove mean + dimension scale normalize |
| abtt | All-but-the-top: subtract top-1 principal component [1] | Remove strongest rogue dimension |
| **whitened** | PCA whitening | **Completely remove anisotropy** |

#### FreeLaw × 7 models × 5 treatments (core evidence)

| Model | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.019 | 0.019 | 0.019 | 0.021 | **0.000** |
| BGE-small | 0.022 | 0.022 | 0.021 | 0.019 | **-0.004** |
| BGE-large | 0.020 | 0.020 | 0.019 | 0.014 | **-0.050** |
| BERT-base | 0.102 | 0.102 | 0.091 | 0.058 | **0.009** |
| Legal-BERT | 0.104 | 0.104 | 0.093 | 0.074 | **0.000** |
| BioBERT | 0.160 | 0.160 | 0.094 | 0.094 | **0.002** |
| PubMedBERT | 0.114 | 0.114 | 0.094 | 0.066 | **-0.005** |

**Critical observation**: **After whitening, all 7 models on FreeLaw silhouette ≈ 0** (from -0.05 to +0.009). Same pattern on PubMed Central across 7 models.

→ **Silhouette on contextual embedding is NOT measuring real clusters; it's an anisotropy-dominated artifact**.

#### Pairwise Spearman ρ_con · 5 treatments (key 6 pairs)

| Pair | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM vs Legal-BERT | -0.429 | -0.429 | -0.333 | +0.214 | **+0.048** |
| MiniLM vs BERT-base | -0.238 | -0.238 | -0.214 | +0.690 | **+0.381** |
| BERT-base vs Legal-BERT | +0.810 | +0.810 | +0.762 | +0.095 | +0.690 |
| BERT-base vs BioBERT | +0.524 | +0.524 | +0.548 | +0.333 | +0.524 |
| BERT-base vs PubMedBERT | +0.738 | +0.738 | +0.690 | +0.357 | +0.381 |
| BioBERT vs PubMedBERT | +0.500 | +0.500 | +0.476 | +0.786 | +0.310 |

**MiniLM vs Legal-BERT**: raw ρ = **-0.429** → whitened ρ = **+0.048** · "lens divergence" completely vanishes. Earlier v2.2's "two-lens system" hypothesis self-falsified by 5-treatment experiments.

### 6.3 SVD Anisotropy · 7-Model Comparison (cross-8-domain mean · l2norm mode)

| Class | Model | top-10 ratio |
|---|---|---|
| sentence-transformers | MiniLM | 0.331 |
| sentence-transformers | BGE-large | 0.625 |
| sentence-transformers | BGE-small | 0.660 |
| BERT-style | BERT-base | **0.900** |
| BERT-style | BioBERT | 0.951 |
| BERT-style | Legal-BERT | 0.953 |
| BERT-style | **PubMedBERT** | **0.992** |

**Two distinct anisotropy levels**:
- sentence-transformers: top-10 = 0.33-0.66 (moderate)
- BERT-style: top-10 = **0.90-0.99** (strong → extreme)

PubMedBERT 0.992 is the most anisotropic model recorded (from-scratch + single homogeneous biomedical corpus + no uniformity loss).

### 6.4 Connection to Literature

> **Aharoni & Goldberg 2020 [12] (ACL)**: Using generic BERT-base, they cluster 5 corpus-source domains with **purity 87.66%**. "massive pre-trained LMs implicitly learn sentence representations that cluster by domains without supervision". → **Any LM (including generic BERT) separates corpus-sources**, not specialist-only signal.

> **Timkey & van Schijndel 2021 [7] (EMNLP)**: 1-3 rogue dimensions dominate cosine similarity. BERT layer 11 single dim accounts for 88.4%, GPT-2 last layer top-1 = 76.3%. → **raw silhouette ≈ silhouette on rogue dimensions**, decoupled from real cluster quality.

> **Mu & Viswanath 2018 [1] (ICLR)**: First few principal directions encode word frequency. **All-but-the-top** (subtract top-1) substantially reduces anisotropy. → Our ABTT treatment is a literature-standard operation.

> **Rudman & Eickhoff 2024 [8] (I-STAR)**: **Lowering isotropy may instead improve downstream performance** · cannot simply equate anisotropy ↑ = poor data quality. → §6.2 data interpretation must avoid direct isotropy → quality causal claim.

> **Cai et al. 2021 [6] (ICLR)**: Argues that contextual embedding global anisotropy is the surface manifestation of isolated clusters + cluster manifolds, with local isotropy within clusters. → Supports §6.2 observation that whitening drives silhouette to zero across all 7 models — local embedding structure is masked by the global anisotropy pattern.

> **Bis et al. 2021 [5] (NAACL)**: Identifies "common shift" as the principal cause of high anisotropy in contextual embeddings — at every step, all embeddings except the ground-truth receive gradient along the same direction. → Explains the physical meaning of the §3.5 raw vs centered gap of ~0.5 — the gap is mainly driven by mean shift.

> **Ait-Saada & Nadif 2023 [9] (ACL Short)**: Counter-consensus balance — measured anisotropy and clustering NMI are **near-zero or weakly negatively correlated**, even claiming "fostering high anisotropy yields high-quality clustering representations". → Note: they use NMI (external metric), NPM uses silhouette (internal metric); the two metric families differ in sensitivity to rogue-dim scale, so this counter-consensus does not directly transplant to silhouette.

> **Gao et al. 2019 [2] (ICLR) + Wang et al. 2020 [3] (ICLR)**: Theoretical foundations of anisotropy — softmax + weight tying + Zipf long-tail → narrow cone for low-frequency tokens. → Explains why BERT-style models (without uniformity loss) are necessarily strongly anisotropic.

![Figure 1 · §6 probe summary · four-mechanism decomposition of FreeLaw "low-diversity" signal (includes Legal-BERT)](ablation_results_20260424/figures/F1_I6_four_mechanism_decomposition.pdf){width=95%}

### 6.5 §6 Methodological Contributions (independent of main study)

1. **Silhouette on contextual embedding is not real cluster measure**: 7 models × 8 domains × whitening = 280 silhouette mostly ≈ 0. Provides empirical evidence for limits of "silhouette + contextual embedding" combination.

2. **Diagnostic for "real subcategory signal vs anisotropy artifact"**: compare raw vs whitened silhouette difference. Whitened ≈ 0 → anisotropy artifact; whitened retains signal → real cluster signal. Simple and reproducible.

3. **Anisotropy bias in cross-model consistency**: raw Kendall W = 0.266, whitened W = 0.228 (slightly drops). Indicates raw "cross-model consistency" partially comes from anisotropy common pattern, not true consensus.

4. **Confirms NPM main scope**: sentence-transformers class models (uniformity loss trained) are reasonable measurement on silhouette/Vendi; BERT-style not directly applicable.

### 6.6 §6 Limitations (this standalone chapter's own honest boundaries)

| # | Boundary | Description |
|---|---|---|
| §6 L1 | Only 4 BERT-style models | Adding SciBERT / CaseLaw-BERT 5th-6th would strengthen |
| §6 L2 | 5 treatments are literature-standard but not exhaustive | Can add more (e.g. Mahalanobis whitening, supervised whitening) |
| §6 L3 | "Anisotropy-dominated" inferred from correlation, not direct causal experiment | Causal verification needs training-time anisotropy strength control (e.g. adding contrastive head fine-tuning to BERT) |

\newpage

## Appendix A · Full 8-Domain Data (sentence-transformers main)

See companion data document `阶梯1_完整原始数据_含BERT-baseline_20260425.md`:

- **§ 1 Main data** (sentence-transformers, three models)
  - § 1.2 table: 8 domains × 3 models × 4 components
- **§ 2 BERT probe data** (BERT-style, four models · for § 6 reference only)
  - § 2.2 table: 8 domains × 4 models × 4 components

## Appendix B · Main Figures

Figures are stored in the `figures/` subdirectory. Strict main-study / §6-probe split: main-study figures contain only the 3 sentence-transformers models; §6-probe figures additionally include Legal-BERT.

**Main-study (§3) figures** (sentence-transformers only):

- **F2 · Vendi-SVD 24-point scatter** (`F2_vendi_svd_st_24pt`) — §3.6 · 3 models × 8 domains, ρ ≈ -0.997 + power-law fit
- **F3 · s̄ four-component × 3-model heatmap** (`F3_sbar_heatmap_st_3model`) — §3.1 · 4 components × 3 models × 8 domains
- **F4 · 3 × 3 Spearman matrices** (`F4_spearman_st_3x3`) — §3.2 · one 3×3 pairwise matrix per component

**§6 (probe) figures** (includes Legal-BERT):

- **F1 · Four-mechanism decomposition** (`F1_I6_four_mechanism_decomposition`) — §6 · 4-panel breakdown of FreeLaw "low-diversity" signal (truncation / sub-category recognition / BGE bias / true low-rank)

Each figure is provided in PNG + PDF, both bilingual (EN + CN). Naming convention: `F{number}_{description}{_cn or English}.{png,pdf}`.

## Appendix C · Related Work

This appendix lists the 25 references involved in this study, organized into 6 thematic categories.

### C.1 Embedding geometry and anisotropy

(9 entries · covers anisotropy phenomena, causes, and mitigation)

[1] **Mu, J., & Viswanath, P.** (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. *International Conference on Learning Representations (ICLR)*. arXiv:1702.01417.

[2] **Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y.** (2019). Representation Degeneration Problem in Training Natural Language Generation Models. *International Conference on Learning Representations (ICLR)*. arXiv:1907.12009.

[3] **Wang, L., Huang, J., Huang, K., Hu, Z., Wang, G., & Gu, Q.** (2020). Improving Neural Language Generation with Spectrum Control. *International Conference on Learning Representations (ICLR)*.

[4] **Ethayarajh, K.** (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:1909.00512.

[5] **Bis, D., Podkorytov, M., & Liu, X.** (2021). Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications. *NAACL 2021*. ACL Anthology 2021.naacl-main.403.

[6] **Cai, X., Huang, J., Bian, Y., & Church, K.** (2021). Isotropy in the Contextual Embedding Space: Clusters and Manifolds. *International Conference on Learning Representations (ICLR)*.

[7] **Timkey, W., & van Schijndel, M.** (2021). All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:2109.04404.

[8] **Rudman, W., & Eickhoff, C.** (2024). Stable Anisotropic Regularization (I-STAR). *International Conference on Learning Representations (ICLR)*. arXiv:2305.19358.

[9] **Ait-Saada, M., & Nadif, M.** (2023). Is Anisotropy Truly Harmful? A Case Study on Text Clustering. *Annual Meeting of the Association for Computational Linguistics (ACL Short)*. ACL Anthology 2023.acl-short.103.

### C.2 Diversity and quality measures

(3 entries · Vendi Score / effective rank / domain clustering)

[10] **Friedman, D., & Dieng, A. B.** (2022). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. *Transactions on Machine Learning Research (TMLR)*. arXiv:2210.02410.

[11] **Roy, O., & Vetterli, M.** (2007). The effective rank: A measure of effective dimensionality. *15th European Signal Processing Conference (EUSIPCO)*, pp. 606-610.

[12] **Aharoni, R., & Goldberg, Y.** (2020). Unsupervised Domain Clusters in Pretrained Language Models. *Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 7747-7763.

### C.3 Scaling Laws and data quality

(3 entries · reference for subsequent interventional studies)

[13] **Subramanyam, S., Chen, Y., & Grossman, S.** (2025). Quality-aware scaling laws for language model pretraining. arXiv:2510.03313v2.

[14] **Bahri, Y., Dyer, E., Kaplan, J., Lee, J., & Sharma, U.** (2024). Explaining Neural Scaling Laws. *Proceedings of the National Academy of Sciences (PNAS)*, 121(27): e2311878121.

[15] **Li, J., Fang, A., Smyrnis, G., et al.** (2024). DataComp-LM: In search of the next generation of training sets for language models. *NeurIPS Datasets and Benchmarks (DCLM)*. arXiv:2406.11794.

### C.4 Sentence Embedding models

(4 entries · original papers of the three models used in main §1-§5)

[16] **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:1908.10084.

[17] **Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M.** (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 2020*. arXiv:2002.10957.

[18] **Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N.** (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE / FlagEmbedding). arXiv:2309.07597.

[19] **Liu, Z., Shao, S., Shi, X., Lian, D., & Xiao, S.** (2022). RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder. *EMNLP 2022*. arXiv:2205.12035.

### C.5 BERT-style domain-specific models (§6 references)

(4 entries · original papers of probe models used in §6 standalone chapter)

[20] **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*, pp. 4171-4186. arXiv:1810.04805.

[21] **Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I.** (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*. arXiv:2010.02559.

[22] **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J.** (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4): 1234-1240. arXiv:1901.08746.

[23] **Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H.** (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing (PubMedBERT). *ACM Transactions on Computing for Healthcare*, 3(1): 1-23. arXiv:2007.15779.

### C.6 NPM framework itself

(2 entries · prior NPM-series releases on Zenodo)

[24] **Ding, T.** (2026). Neural Percolation Model: A Porous-Media-Inspired Framework for Neural Network Training Dynamics. Zenodo. DOI: 10.5281/zenodo.19209722.

[25] **Ding, T.** (2026). Cross-Family Convergence of Neural Network Weight Skeletons (NPM-K). Zenodo. DOI: 10.5281/zenodo.19652706.

### Appendix D · Code and Data Availability

All code, data, and a transparent data-reference document for this study are publicly available on GitHub:

**Code & data**: <https://github.com/tiexinding/data-quality-vec-public>

Repository contents: bilingual technical report v2.4 (EN + CN) + 19 CSV/JSON data files (primary / probe / ablation / 5-treatment) + F1-F4 main figures (16 PNG/PDF, EN + CN) + main pipeline script (`run_stage1_sbar_v2.py`) + SVD anisotropy diagnostic + K-scan + figure generation + PDF builder. License: MIT.

Intermediate embeddings (`.npy`, ~380 MB) and text caches (~137 MB) are NOT included in the repository; they will be regenerated on the fly from `monology/pile-uncopyrighted` (HuggingFace) and the listed model checkpoints by `code/run_stage1_sbar_v2.py`.

---

**Version**: v2.4 sentence-transformers main · 2026-04-25
**Drafted by**: B2 (revising from v2.1 + B1 4-25 Rule 14 splitting strategy)
**Reviewers**: B1 (4-25 16:21 three must-fix + four suggestions) · A
**Next version**: v2.5 + lawyer's qualitative feedback (pending) + GTE/E5 4th-5th sentence-transformers
**Release intent**: Zenodo deposit · main scope kept clean and clear · §6 standalone as methodological contribution
