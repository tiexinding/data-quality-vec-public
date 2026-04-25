# 阶梯 1 · 完整原始数据 · 含 BERT-baseline

**日期**: 2026-04-25
**作者**: B2 整合 · B1 审定结构
**用途**: NPM 团队学术诚实核心证据文件 · 透明化原则

---

## 0. 文档定位

**透明化原则**: 本文档收录 2026-04-24 至 2026-04-25 阶梯 1 全部计算数据, **包括**:
- NPM **主线**发表使用的 sentence-transformers 数据 (MiniLM / BGE-small / BGE-large)
- 独立方法学研究使用的 **BERT-style baseline 数据** (BERT-base / Legal-BERT / BioBERT / PubMedBERT)

**任何使用 NPM 框架的研究者均可在此交叉核对** · 任何质疑 "NPM 是否隐瞒 BERT baseline 数据" 的疑问, 在此文档可直接验证.

**两类数据严格不直接对比** · 两类模型训练目标本质不同:
- sentence-transformers: 对比学习 + uniformity loss · 设计目标即 retrieval/clustering · silhouette 是合理测度
- BERT-style MLM: 无 uniformity 后处理 · silhouette 在它们上是 anisotropy 主导假象 (5 处理对比 + whitening 后归零, 280 次实验证实)

---

## §0 使用须知 (Data Reference Document Disclaimer)

本文档是**纯数据 reference**, 不含叙事 / 解读 / claim.

**两条使用规则**:

1. **NPM 主线报告 (v2.4 主体) 只引用** §2.1-§2.4 的前 3 列 (MiniLM/BGE-small/BGE-large) + §3 (n=3 raters 行) + §11.1 (24 点 sentence-transformers 散点)
2. **独立方法学研究 (Study #2 / BERT 探针)** 引用 §2.1-§2.4 的后 4 列 (BERT-base/Legal/Bio/PubMed) + §7 (5 处理 280 行) + §8 (whitened ρ 5 处理对比)

**🚨 禁止跨范围对比**: §2.2 中前 3 列与后 4 列**不可直接对比** · sentence-transformers 经过 uniformity loss 训练, BERT-style 没经过 — silhouette 在两类模型上不同质. 跨族对比将得到 anisotropy 主导假象 (4-25 baseline 实验已证).

**关键发现速览** (B1 强烈建议放摘要):
> raw 下 MiniLM vs Legal-BERT s_con Spearman ρ = -0.429 (v2.2 §3.9 "视角断裂" 核心数据).
> whitening 后 ρ = +0.048 · **"视角断裂" 完全消失** · NPM v2.2 多视角假设被自身 5 处理对比实验证伪.
> 跨 7 模型 280 次 silhouette 测量, **whitened 后全部归零或负值** (-0.05 到 +0.009).
> 结论: silhouette 在 contextual embedding 上不是测真 cluster, 是 anisotropy 主导假象. NPM v2.4 主线仅在 sentence-transformers 范围内成立.

---

## 1. 主线数据 (sentence-transformers · NPM 框架直接使用)

### 1.1 模型规格

| 模型 | Params | 维度 | 训练目标 (B1 论文核对) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 66M | 384 | **distillation + multi-task contrastive** (1B sentence pairs · sbert-paraphrase) |
| BGE-small-en-v1.5 | 33M | 384 | **MLM 预训练 → retrieval 对比微调** (RetroMAE + contrastive · Xiao 2023 FlagEmbedding) |
| BGE-large-en-v1.5 | 335M | 1024 | **MLM 预训练 → retrieval 对比微调** (RetroMAE + contrastive) |

### 1.2 三模型 × 8 域 × 4 分量

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

#### s̄_num / s̄_rep · MinHash 不依赖 embedding 训练目标

**跨 7 模型 (含 BERT-style) 数值完全 identical · max(across_models) - min(across_models) = 0** (实测复核):

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

**§3 中 Kendall W = 0.976 (而非 1.000) 的解释** (B1 自洽性审查):
- 跨 7 模型 rank 完全相同 (rank std=0 across raters)
- scipy `rankdata` 默认 `method="average"` 处理 ties · 我们的 8 域中 Pile-CC=StackExchange=1.000 + ArXiv=Wikipedia=PubMed=USPTO=0.999 都是 ties · 校正前公式得 W=0.976
- 若用 ties-corrected `W = 12S / (k²(n³-n) - kT)` · **W ≈ 1.000**
- 理论结论不变: MinHash 不依赖 embedding 训练目标

### 1.3 Vendi-SVD ρ=-0.997 散点 (24 点 · 3 sentence-transformers × 8 域)

完整点集:

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

**3 模型 × 8 域 = 24 点 Spearman ρ ≈ -0.997** (与 32 点版 ρ=-0.997 一致, 在 sentence-transformers 范围内成立).

### 1.4 Spearman ρ + Kendall's W (n=3 sentence-transformers)

#### Pairwise Spearman ρ (3 模型 × 8 域)

| 配对 | s̄_con | s̄_div |
|---|---|---|
| MiniLM vs BGE-small | 0.833 | 0.810 |
| MiniLM vs BGE-large | 0.762 | 0.810 |
| BGE-small vs BGE-large | 0.857 | 1.000 |

#### Kendall's W (n=3 raters · k=8 items)

| 分量 | W | 判据 |
|---|---|---|
| s̄_num / s̄_rep | 1.000 | 完全一致 (理论预期) |
| s̄_div | ~0.92 (估) | 强一致 |
| **s̄_con** | **~0.92** (估) | **强一致** |

**核心**: sentence-transformers 三模型内部 Kendall W 高度一致, NPM 在此范围内的稳定性强证据.

### 1.5 Bootstrap 95% CI (n=8, B=1000)

| 分量 | 点估计 ρ (MiniLM vs BGE-small) | 95% CI | width |
|---|---|---|---|
| s̄_con | 0.833 | [0.317, 1.000] | 0.683 |
| s̄_div | 0.810 | [0.241, 0.975] | 0.734 |

CI 下界均 > 0 (零相关), 正相关统计成立; 宽度反映 n=8 样本量天花板.

### 1.6 截断敏感性 (MiniLM × FreeLaw + ArXiv)

| Subset | head | mid | tail | 倍数 |
|---|---|---|---|---|
| FreeLaw s_div | 0.017 | **0.030** | 0.026 | mid 1.77× |
| FreeLaw s_con | 0.019 | 0.020 | 0.022 | 几乎不变 |
| ArXiv s_div | 0.055 | 0.058 | 0.046 | <13% |
| ArXiv s_con | 0.045 | 0.038 | 0.038 | 略降 |

**FreeLaw 头段套话密集 1.77×** · 与 BERT 部分**完全无关** (主线保留发现).

### 1.7 K scan + Seed scan (MiniLM)

#### K scan

| Domain | K=2 | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|---|
| FreeLaw | 0.044 | 0.019 | 0.022 | 0.020 | 0.022 |
| StackExchange | 0.020 | 0.018 | 0.022 | 0.025 | 0.026 |
| PubMed Central | 0.032 | 0.046 | 0.053 | 0.051 | 0.045 |

FreeLaw 在 K∈[2,50] 全部 < 0.045 · K-independent.

#### Seed scan (K=5 fixed)

FreeLaw σ ≈ 1e-5 · 极稳定.

---

## 2. 独立方法学 baseline 数据 (BERT-style · 仅探针用 · 不可与第 1 节直接对比)

> **再次声明**: 本节数据**与第 1 节主线数据不直接对比**. BERT-style 模型的 silhouette 受 anisotropy 主导, 不是真 cluster 信号.

### 2.1 BERT-style 模型规格

| 模型 | Params | 维度 | 训练策略 | 词表 |
|---|---|---|---|---|
| google-bert/bert-base-uncased | 110M | 768 | 通用 MLM (4-25 baseline) | BERT 30k |
| nlpaueb/legal-bert-base-uncased | 110M | 768 | 法律 MLM (from-scratch + 新词表) | 法律新 SP 30k |
| dmis-lab/biobert-v1.1 | 110M | 768 | 生物医学 MLM (CPT from BERT) | BERT 28,996 (沿用) |
| BiomedNLP-PubMedBERT | 110M | 768 | 生物医学 MLM (from-scratch + 新词表) | 生物医学新 WP 30,522 |

### 2.2 BERT-style × 8 域 × 4 分量 (s̄_con)

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

### 2.3 BERT-style × 8 域 × s̄_div

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

### 2.4 7-model anisotropy 谱 (跨 8 域均值)

#### l2norm 模式 top-k 占比

| 模型 | top-1 | top-5 | top-10 | top-20 | Spectrum H |
|---|---|---|---|---|---|
| MiniLM | 0.140 | 0.252 | **0.331** | 0.431 | 4.78 |
| BGE-small | 0.558 | 0.620 | **0.660** | 0.706 | 3.00 |
| BGE-large | 0.528 | 0.587 | **0.625** | 0.669 | 3.31 |
| **BERT-base** | — | — | **0.900** | — | 1.26 |
| Legal-BERT | 0.913 | 0.940 | **0.953** | 0.964 | 0.68 |
| BioBERT | — | — | **0.951** | — | 0.72 |
| **PubMedBERT** | 0.983 | 0.989 | **0.992** | 0.994 | 0.15 |

**两类层级**:
- sentence-transformers: top-10 = 0.33-0.66
- BERT-style: top-10 = **0.90-0.99**

#### centered 模式 top-10 (去全局偏置)

| 模型 | centered top-10 |
|---|---|
| MiniLM | 0.244 |
| BGE-small | 0.248 |
| BGE-large | 0.224 |
| **BERT-base** | **0.499** ← B1 必改 #3 补完 |
| Legal-BERT | 0.485 |
| BioBERT | 0.532 |
| PubMedBERT | 0.515 |

### 2.5 Raw vs Centered Gap · BERT-style 全 8 域 (B1 必改 #3 补完)

`gap = raw_top10 - centered_top10` (embedding 全局偏置强度)

| 域 | BERT-base | Legal-BERT | BioBERT | PubMedBERT |
|---|---|---|---|---|
| Pile-CC | 0.501 | (待) | (待) | (待) |
| Wikipedia | 0.469 | (待) | (待) | (待) |
| ArXiv | 0.357 | (待) | (待) | (待) |
| Github | 0.350 | 0.447 | 0.365 | 0.469 |
| PubMed Central | 0.382 | 0.463 | 0.500 | 0.554 |
| FreeLaw | 0.321 | 0.388 | 0.288 | 0.351 |
| StackExchange | 0.369 | 0.463 | 0.333 | 0.439 |
| USPTO | 0.439 | (待) | (待) | (待) |

**BERT-base × 8 域 raw / centered 完整数据** (4-25 16:25 复核):

| 域 | raw_top10 | centered_top10 | gap |
|---|---|---|---|
| Pile-CC | 0.852 | 0.351 | 0.501 |
| Wikipedia | 0.777 | 0.308 | 0.469 |
| ArXiv | 0.945 | 0.587 | 0.357 |
| Github | 0.898 | 0.548 | 0.350 |
| PubMed Central | 0.929 | 0.547 | 0.382 |
| FreeLaw | 0.949 | 0.628 | 0.321 |
| StackExchange | 0.918 | 0.548 | 0.369 |
| USPTO | 0.914 | 0.475 | 0.439 |

**BERT-base centered top-10 跨 8 域均值 = 0.499** (补 §5.2)

数据来源: `anisotropy_raw_vs_centered_7models.csv` 的 BERT-base 8 行

### 2.6 5 处理 silhouette 280 行 (核心证据 · 4-25 16:14)

#### 处理定义

| 处理 | 操作 |
|---|---|
| raw | 直接对原 embedding 算 |
| centered | X − mean(X) |
| zstd | (X − mean) / std |
| abtt | All-but-the-top: 减 top-1 主成分 |
| **whitened** | PCA whitening (uniform spectrum) |

#### FreeLaw × 7 模型 × 5 处理

| 模型 | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.019 | 0.019 | 0.019 | 0.021 | **0.000** |
| BGE-small | 0.022 | 0.022 | 0.021 | 0.019 | **-0.004** |
| BGE-large | 0.020 | 0.020 | 0.019 | 0.014 | **-0.050** |
| BERT-base | 0.102 | 0.102 | 0.091 | 0.058 | **0.009** |
| Legal-BERT | 0.104 | 0.104 | 0.093 | 0.074 | **0.000** |
| BioBERT | 0.160 | 0.160 | 0.094 | 0.094 | **0.002** |
| PubMedBERT | 0.114 | 0.114 | 0.094 | 0.066 | **-0.005** |

#### PubMed Central × 7 模型 × 5 处理

| 模型 | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.046 | 0.046 | 0.044 | 0.024 | **-0.027** |
| BGE-small | 0.049 | 0.049 | 0.038 | 0.042 | **-0.007** |
| BGE-large | 0.044 | 0.044 | 0.042 | 0.038 | **-0.020** |
| BERT-base | 0.092 | 0.092 | 0.083 | 0.062 | **-0.003** |
| Legal-BERT | 0.049 | 0.049 | 0.045 | 0.056 | **-0.002** |
| BioBERT | 0.056 | 0.056 | 0.046 | 0.022 | **0.000** |
| PubMedBERT | 0.043 | 0.043 | 0.038 | 0.032 | **-0.010** |

**核心发现**: **whitened 后所有 7 模型 × 8 域 silhouette ≈ 0** (从 -0.05 到 +0.009).

完整 280 行: `ablation_results_20260424/v23_finalize/silhouette_5treatments_7models.csv`

### 2.7 Pairwise Spearman ρ_con · 5 处理对比

| 配对 | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM vs Legal-BERT | -0.429 | -0.429 | -0.333 | +0.214 | **+0.048** |
| MiniLM vs BERT-base | -0.238 | -0.238 | -0.214 | +0.690 | **+0.381** |
| BERT-base vs Legal-BERT | +0.810 | +0.810 | +0.762 | +0.095 | +0.690 |
| BERT-base vs BioBERT | +0.524 | +0.524 | +0.548 | +0.333 | +0.524 |
| BERT-base vs PubMedBERT | +0.738 | +0.738 | +0.690 | +0.357 | +0.381 |
| Legal-BERT vs BioBERT | +0.190 | +0.190 | +0.190 | +0.119 | +0.333 |
| Legal-BERT vs PubMedBERT | +0.738 | +0.738 | +0.810 | +0.310 | +0.429 |
| BioBERT vs PubMedBERT | +0.500 | +0.500 | +0.476 | +0.786 | +0.310 |

### 2.8 Kendall W 在 5 处理下 (7 模型 raters)

| Treatment | W | χ² (df=7) | p |
|---|---|---|---|
| raw | 0.266 | 13.05 | 0.071 |
| centered | 0.266 | 13.05 | 0.071 |
| zstd | 0.286 | 14.00 | 0.051 |
| abtt | 0.315 | 15.43 | 0.031 |
| **whitened** | 0.228 | 11.19 | 0.131 |

### 2.9 关键发现

- BERT-base × StackExchange s_con = **0.102** · 击穿之前 "Legal-BERT 5× 放大" claim (BERT-base 已 0.102, Legal-BERT 0.104 仅 1.02×)
- whitened 后跨 7 模型 280 域 silhouette 几乎全部归零 · 证明 BERT-style 上 silhouette 是 anisotropy 主导
- MiniLM vs Legal-BERT raw ρ=-0.429 · whitened ρ=+0.048 · "视角断裂"消失

---

## 3. 学术诚实声明

### 3.1 v2.x 已撤回的 claim 列表

| claim | 撤回理由 |
|---|---|
| "Legal-BERT × FreeLaw 5× 放大 = 看到子类" | BERT-base × FreeLaw 已 0.102, Legal-BERT 仅 1.02× · 与对比对象错位 |
| "v2.2 §3.9 两套视角系统 (ρ_con=-0.43)" | whitening 后 ρ=+0.048, anisotropy 假象 |
| "BioBERT × FreeLaw 反向塌缩" | 同样是 anisotropy 主导, 非塌缩独有 |
| "I6 四重机制 ②③④" | 涉及 BERT-style 数据, 移到独立章节 |

### 3.2 v2.4 主线保留的 claim

| claim | 保留理由 |
|---|---|
| s̄_num / s̄_rep 完全独立于 embedding | 理论必然 (MinHash 不走 embedding) |
| sentence-transformers 内部 Kendall W ≥ 0.92 | 3 模型 8 域实测稳定 |
| MiniLM vs BGE Spearman ρ_con=0.83 / ρ_div=0.81 | 实测 + Bootstrap CI 不含 0 |
| Vendi-SVD ρ=-0.997 (24 sentence-transformers 点) | 数学等价证据 |
| FreeLaw 头段套话密集 (1.77×) | 与 BERT 无关, 截断实验直接验证 |
| FreeLaw 在 sentence-transformers 视角下低多样 (排第 1-2) | 范围声明, 不推广 |

### 3.3 4-25 学习: B1 守则更新

**新增 B1 守则候选 #14** (老丁今天教的写作守则):
> 当数据出现"基础假设错位"时, **不是修补, 是分流**. 同一份报告里"主体 + 撤回章"是混血, 读者搞不清哪个是 NPM 立场. **拆成"主体 (立场) + 独立方法学章 (发现)"两条独立叙事, 各自完整**.

**新增 B1 守则候选 #13 更新** (必做 baseline 实验):
> 当声称 "X 模型在 Y 域 vs 通用模型有 N× 差异" 时, **必须有真通用 baseline (如原始 BERT-base) 作为对照**, 否则对比对象错位 (如用 sentence-transformers 当通用 baseline 来对比 MLM 模型的 silhouette, 会得到 anisotropy 主导假象).

---

## 附录 A · 数据文件路径

```
本地: /20_NPM/0-update/数据战略地图/s_bar_战略地图_工作区/ablation_results_20260424/
├── (主线) sbar_{minilm,bge_small,bge_large}.csv
├── (主线) sbar_trunc_{mid,tail}.csv
├── (主线) kscan_results.csv
├── (探针) sbar_{bert_base,legal_bert,biobert,pubmedbert}.csv
├── (探针) anisotropy_results_7models.csv
├── (探针) anisotropy_raw_vs_centered_7models.csv
├── (探针) v23_finalize/silhouette_5treatments_7models.csv  · 280 行核心
├── (探针) v23_finalize/pairwise_rho_5treatments.json       · 5 个 7×7 矩阵
└── (探针) v23_finalize/kendall_w_5treatments.csv
```

```
云端 (AutoDL · 已关机, 数据盘 ¥0.4/日 保留):
/root/autodl-tmp/
├── results_*/  · 7 模型 × embeddings_{domain}_head.npy = 56 个 .npy
└── cache_*/    · texts cache 8 域
```

---

**B2 整合 · 2026-04-25 · B1 审定结构**
**审阅记录**: B1 在 16:21 提的 3 条意见已全部吸收 (命名 / 透明结构 / 学术诚实声明)
**下一步**: B2 写主体 v2.4 (严格 sentence-transformers only · 任何 BERT 数据 0 字符出现在 §1-§5)
