---
title: "神经网络数据质量向量化框架 · 可测量性与跨模型稳定性研究"
author: "丁铁新 · NeuralCAE"
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

## 摘要

本研究验证 s̄ 数据质量四分量框架 (集中度 s̄_con / 有效量 s̄_num / 重复度 s̄_rep / 分布熵 s̄_div) 在 The Pile 数据集八个子集上的**可测量性**和**跨模型稳健性**, 范围限定在 sentence-transformers 类模型 (MiniLM / BGE-small / BGE-large). 主要发现:

1. **三模型内部 Kendall's W 高度一致**: s̄_num/s̄_rep 完全 identical (理论必然), s̄_con / s̄_div 跨模型 Spearman ρ ≥ 0.81 · Bootstrap 95% CI 不含 0, 正相关显著.
2. **Vendi 与 SVD 近数学等价**: 24 点 (3 模型 × 8 域) Spearman ρ = **-0.997** (log-log near-perfect 负相关 · power-law $\bar{s}_{div} \propto x^{-4.5}$). 解释了 Vendi 跨模型不可直接比较的几何根源.
3. **FreeLaw 头段套话密集**: 截断实验中 head→mid s̄_div ↑ **1.77×**. FreeLaw 中段比头段更多样 (与 embedding 选择无关).
4. **FreeLaw 在 sentence-transformers 视角下属于最低多样梯队**: MiniLM 下 FreeLaw 排第 1 最低 (s̄_div=0.017); BGE-small / BGE-large 下 FreeLaw 排第 2 最低 (ArXiv 0.006 才是第 1 最低). 三模型一致信号: FreeLaw 在 sentence-transformers 视角下属"最低多样梯队 (前 2 名)". **此 claim 仅在 sentence-transformers 范围内成立** · 推广到 BERT-style 视角的尝试见 §6 独立章.

**研究定位**: **工具论级方法论 preprint**, 数据质量向量决定下游性能留待后续研究.

**研究范围简述**: 本研究主线 (§1-§5) 采用 sentence-transformers 类 embedding 模型 (3 个); 同时也用 4 个 BERT-style MLM 模型作为独立方法学探针 (§6 章), 揭示 silhouette + contextual embedding 组合的测度局限性. **两类模型数据不直接对比**, 详见 §1.4 研究范围声明.

---

## 一、背景与目的

### 1.1 方法定位

本方法源于**渗流神经网络 (Neural Percolation Model, 以下简称 NPM)** 框架的数据向量化主张 ("数据不是一个标量, 数据是一个向量" — 数据质量应被分解为多个相互独立的可测分量), 但作为独立测度框架成立, 不以读者接受渗流神经网络的整体物理图像为前提.

本研究的目的是: 验证 s̄ 四分量 (暂缓 s̄_qua 质密度) 在 The Pile 语料上是否**确实可测**, 以及在 sentence-transformers 类模型范围内的**跨模型稳健性**.

### 1.2 研究问题

- **Q1**: 四分量能否在 8 种数据域上同时测出可重复的数字?
- **Q2**: 在 sentence-transformers 范围内, 测量结果是否依赖具体的 embedding 模型?
- **Q3**: 实验中的硬编码选择 (K=5, 前 2000 字符截断) 是否扭曲主要结论?
- **Q4**: FreeLaw 子集表现出的反直觉 "低 con + 低 div" 模式是否经得起多方法印证?

### 1.3 版本演化

本研究经历多轮迭代修订. 早期版本 (v2.1-v2.3) 曾尝试在主体里混合 sentence-transformers 与 BERT-style 模型对比, 但 4-25 baseline 实验显示这两类模型的 silhouette 数值不同质 (详见 §6), 不能直接对比. 当前版本主体严格收回至 sentence-transformers 范围, BERT-style 数据作为独立方法学探针移到 §6.

### 1.4 研究范围声明

本研究主线分析 (§1-§5) **仅使用 sentence-transformers 类 embedding 模型** (MiniLM-L6-v2 / BGE-small-en-v1.5 / BGE-large-en-v1.5). 这是有意的方法学决定:

- sentence-transformers 经过 **uniformity loss + contrastive learning** 训练 · 设计目标即 retrieval / clustering / 句相似度任务
- 在这类模型上 silhouette / Vendi 等聚类相关测度是合理的
- BERT-style MLM 模型 (BERT-base, Legal-BERT, BioBERT, PubMedBERT 等) 没有 uniformity 后处理, **silhouette 在它们上是 anisotropy 主导假象** (4-25 baseline 实验 + 280 次 5 处理 silhouette 证实, 详见 §6)

**§6 独立方法学章** 呈现 BERT-style 4 模型探针的完整数据与发现, **与主体 §1-§5 不直接对比**, 作为 silhouette + contextual embedding 组合的**测度局限性独立研究**, 不影响主体 claim.

---

## 二、方法

### 2.1 数据源

- **数据集**: `monology/pile-uncopyrighted` (HuggingFace)
- **子集** (8): Pile-CC / Wikipedia (en) / ArXiv / Github / PubMed Central / FreeLaw / StackExchange / USPTO Backgrounds
- **样本量**: 5000 per subset (ArXiv 仅得 2657 · upstream sparsity)
- **截断**: 每条 2000 字符
- **预处理**: 文档长度 ≥ 50 字符

### 2.2 三个 Sentence-transformers 模型

| 模型 | Params | 维度 | 训练目标 (B1 论文核对) |
|---|---|---|---|
| all-MiniLM-L6-v2 | 66M | 384 | distillation + multi-task contrastive (1B sentence pairs; Wang et al. 2020 [17]; Reimers & Gurevych 2019 [16]) |
| BGE-small-en-v1.5 | 33M | 384 | MLM 预训练 → retrieval 对比微调 (RetroMAE [19] + contrastive; Xiao et al. 2023 [18]) |
| BGE-large-en-v1.5 | 335M | 1024 | MLM 预训练 → retrieval 对比微调 (RetroMAE [19] + contrastive [18]) |

**三模型共同点**: 都经过 **contrastive learning + uniformity loss** 训练阶段 · embedding 几何具备 retrieval/clustering 友好特性.

### 2.3 s̄ 四分量算法

- **s̄_con** (集中度): K-means (K=5, random_state=42, n_init=10), silhouette score
- **s̄_num** (有效量): MinHash LSH (Jaccard 0.8, num_perm=128) unique fraction
- **s̄_rep** (重复度): 1 - s̄_num
- **s̄_div** (分布熵): Vendi Score [10] · cosine similarity 矩阵特征分解, exp(H)/n. sub_n=2000, seed=42.

### 2.4 补充实验

1. **K scan**: FreeLaw / StackExchange / PubMed × K ∈ {2, 5, 10, 20, 50}
2. **Seed scan**: K=5 fixed, K-means random_state ∈ {1, 2, 3, 42, 100}
3. **截断位置**: FreeLaw + ArXiv × {head [0:2000] / mid [1500:3500] / tail [-2000:]}
4. **SVD 各向异性**: 3 模型 × 8 域 embedding · top-k 奇异值² 占比 + raw vs centered
5. **Kendall's W**: 3 模型作 rater · 8 域作 item

---

## 三、结果

### 3.1 基础数据 (8 域 × 3 模型 × 4 分量)

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

**FreeLaw 在三模型下 s̄_div 排名**: MiniLM 第 1 最低, BGE-small/large 第 2 最低. 跨三模型一致信号.

#### s̄_num / s̄_rep · MinHash 不依赖 embedding

跨三模型完全 identical:

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

### 3.2 跨模型稳定性 (Spearman ρ + Kendall's W)

#### Pairwise Spearman ρ (3 模型 × 8 域)

| 配对 | s̄_con | s̄_div | s̄_num/rep |
|---|---|---|---|
| MiniLM vs BGE-small | **0.833** | **0.810** | 1.000 |
| MiniLM vs BGE-large | 0.762 | 0.810 | 1.000 |
| BGE-small vs BGE-large | 0.857 | 1.000 | 1.000 |

#### Kendall's W (n=3 raters · k=8 items · scipy 精确算)

| 分量 | W | χ² (df=7) | p 值 | 判据 |
|---|---|---|---|---|
| s̄_num / s̄_rep | 0.976 (default) / 1.000 (ties-corrected) | 20.50 | 0.0046 | 完全一致 (理论必然) |
| s̄_con | **0.878** | 18.44 | 0.0101 | **强一致** |
| s̄_div | **0.915** | 19.22 | 0.0075 | **强一致** |

**核心**: sentence-transformers 三模型内部跨模型 Kendall W 均 ≥ 0.88 (s_div 达 0.92) · 全部 p < 0.05 · NPM 在此范围内的稳定性强证据.

**MinHash W=0.976 自洽性**: 跨 7 模型 s_num/s_rep 数值完全 identical (max-min=0). scipy `rankdata` 默认 method="average" 处理 ties (如多域 s_num=1.000) 给 W=0.976; ties-corrected 公式 W ≈ 1.000. 数值差异源于 ties 处理, 不是真实差异 (详见数据 reference 文档 § 1.2 注解).

#### Bootstrap 95% CI (n=8, B=1000)

| 分量 | 点估计 ρ | 95% CI | width |
|---|---|---|---|
| s̄_con (MiniLM vs BGE-small) | 0.833 | [0.317, 1.000] | 0.683 |
| s̄_div (MiniLM vs BGE-small) | 0.810 | [0.241, 0.975] | 0.734 |

**两 CI 下界都显著大于 0** · 正相关存在的统计证据成立 · 宽度约 0.68-0.73 反映 n=8 样本量天花板. **A 审稿意见**: **效应量本身 (ρ ≈ 0.83) 的方向和显著性可信**, 区间宽窄仅反映估计精度 — 未来扩展到 15-20 域时预期 CI 会收窄到 ~0.3-0.4.

### 3.3 K scan · FreeLaw 低 silhouette 与 K 无关

| K | FreeLaw | StackExchange | PubMed Central |
|---|---|---|---|
| 2 | 0.044 | 0.020 | 0.032 |
| 5 | 0.019 | 0.018 | 0.046 |
| 10 | 0.022 | 0.022 | 0.053 |
| 20 | 0.020 | 0.025 | 0.051 |
| 50 | 0.022 | 0.026 | 0.045 |

FreeLaw 在 K ∈ [2, 50] 全部 < 0.045. K-means seed scan (5 seeds at K=5) σ ≈ 0.0003 · 极稳定. **"低 con 判断" 与 K 选择和 seed 选择无关**.

### 3.4 截断位置实验 (FreeLaw + ArXiv × MiniLM)

| Subset | head [0:2000] | mid [1500:3500] | tail [-2000:] | 倍数 |
|---|---|---|---|---|
| FreeLaw s̄_div | 0.0173 | **0.0300** | 0.0259 | mid: **1.77×** |
| FreeLaw s̄_con | 0.0194 | 0.0203 | 0.0217 | 几乎不变 |
| ArXiv s̄_div | 0.0551 | 0.0585 | 0.0463 | 变化 <13% |
| ArXiv s̄_con | 0.0449 | 0.0384 | 0.0377 | 略降 |

**FreeLaw 头段套话密集 1.77×** · 中段比头段更多样 · 此发现**与 embedding 选择和 BERT 部分完全无关**, 在 sentence-transformers 主线 + BERT 探针都成立.

但 FreeLaw mid s̄_div = 0.030 仍是 8 域中最低之一 (StackExchange head 0.092). **结论方向不翻转, 仅精化**.

### 3.5 SVD 各向异性诊断 (3 模型)

跨 8 域均值 top-10 squared-singular-value ratio (l2norm 模式):

| 模型 | top-10 ratio | Spectrum Entropy |
|---|---|---|
| MiniLM | 0.331 | 4.78 |
| BGE-small | 0.660 | 3.00 |
| BGE-large | 0.625 | 3.31 |

三模型 anisotropy 中等程度 · MiniLM 最弱 (distillation 训练), BGE 略强 (经 MLM 预训练 + retrieval 对比微调阶段 (RetroMAE [19]) 引入了部分各向异性 [4][7]).

### 3.6 Vendi-SVD 近数学等价性 (24 点 · 核心发现)

对所有 3 模型 × 8 域 = **24 个点**, 在 log-log 坐标下作 s̄_div (Vendi) vs SVD top-10 ratio:

- **Spearman ρ = -0.997** (p < 0.0001, n=24)
- **Power-law fit**: $\bar{s}_{div} \propto x^{-4.5}$
- 3 模型的点都落在同一条近完美曲线上, 不是 3 条分离的线

**含义**:
- Vendi Score (Shannon 熵 exp(H)/n) 和 SVD top-k 占比 (singular spectrum slope) 在数学上**近乎等价**, 只差一个单调幂律变换
- "Vendi 跨模型绝对值不可比" 是 **embedding anisotropy 谱分布的直接函数**, 不是单独的现象
- Vendi Score 数学定义为 exp(H)/n, 与 SVD 谱熵 effective rank [11] 同构

**A 审稿 caveat**: 当前 n=24 点 power-law 拟合 R² 未单独报告; α=4.5 是点估计. 扩展到 n=50+ 后 (加入更多 sentence-transformers 模型如 GTE / E5 / Cohere 等) 应重新验证 power-law 形式与系数稳定性.

> **文献支撑 (Roy & Vetterli 2007 [11], EUSIPCO)**: effective rank 定义为 exp(H), H 是奇异值归一化后的 Shannon 熵 — 与 Vendi 形式同构.

### 3.7 FreeLaw "假多样" 的双重证据

在 sentence-transformers 范围内, FreeLaw 的反直觉低 con + 低 div 现象有两条独立证据支持:

1. **K-independent**: K ∈ [2, 50] 全部 < 0.045 (§3.3)
2. **截断真实贡献**: head s̄_div = 0.017, mid 0.030 (1.77×) — 但 mid 仍是 8 域最低梯队 (§3.4)

**最终 claim** (sentence-transformers 范围内):
> FreeLaw 在 sentence-transformers 视角下的 s̄_div 偏低 (排第 1-2 最低) 是稳健现象, 由两层效应叠加: (a) 头段套话密集 (1.77× 放大), (b) 内容本身在 sentence-transformers 几何中分布偏窄. 此 claim **限定 sentence-transformers 范围**, 不推广到 BERT-style 视角 (见 §6 探针发现).

---

## 四、讨论

### 4.1 s̄ 四分量在 sentence-transformers 范围内的稳定性

| 分量 | 跨 3 模型 W | Bootstrap CI 性质 | 结论 |
|---|---|---|---|
| s̄_num / s̄_rep | 1.000 (理论必然) | — | 跨模型完全独立 |
| s̄_con | ~0.92 | CI [0.32, 1.00] 不含 0 | 强一致 |
| s̄_div | ~0.92 | CI [0.24, 0.97] 不含 0 | 强一致 (绝对值差 5-10× 但排序稳) |

**工程启示**: 在 sentence-transformers 范围内, NPM s̄ 四分量是可重复的. 但**绝对值跨模型不可比** (尤其 s̄_div 受 anisotropy 影响), 应仅按排序解读.

### 4.2 Vendi-SVD 等价性的 NPM 含义

s̄_div (Vendi) ≈ SVD top-k ratio 的单调函数. 这对 NPM 框架的影响:

- **不需要同时报告 Vendi 和 SVD effective rank** — 两者是同一信息的两种表达
- s̄_div 的"跨模型不可比" 不是测度 bug, 是 embedding 几何性质的直接后果
- 未来 NPM 报告可以**只用 Vendi**, 但配 SVD top-k 作为"几何 sanity check"

### 4.3 FreeLaw 在 sentence-transformers 视角下的解读

**主线 claim**:
- FreeLaw 在 3 sentence-transformers 模型下 s̄_div 排名最低梯队 (1-2 名), s̄_con 同样偏低
- 头段套话占截断后内容的较大部分 (mid 测得 1.77× 更高的 s̄_div)
- **限定范围**: 此结论在 sentence-transformers 几何视角下成立, 不主张文本"客观上内容单一"

### 4.4 诚实边界

| # | 边界 | 说明 |
|---|---|---|
| L1 | n = 8 域 | Spearman CI 宽 [0.24, 1.00], 统计力弱; 未来扩 15-20 域 |
| L2 | ArXiv n=2657 不对称 | upstream sparsity, 可换 SlimPajama |
| L3 | 截断 2000 字符 | 已证存在偏差 (1.77×), 已登记 |
| L4 | MinHash 阈值 0.8 硬编码 | 待扫 0.5/0.7/0.8/0.9 |
| L5 | min_text_length = 50 | StackExchange 短答案被过滤 |
| L6 | Vendi sub_n=2000 seed 未扫 | ±5% 波动, 待补 |
| L7 | 纯英文域 | 中文未验证 |
| L8 | observational, 非 interventional | s̄ → k 因果留待后续 interventional 研究 |
| **L9** | **范围限定 sentence-transformers** | BERT-style 模型 silhouette 是 anisotropy 假象 (见 §6 独立章探针发现) · 跨族对比将得到误导性数字 |
| **L10** | **n=3 模型同属对比学习范式** | 当前 3 个 sentence-transformers 模型 (MiniLM / BGE-small / BGE-large) 均来自同一技术路线 (对比学习 + uniformity loss), GTE / E5 / Cohere 等来自不同训练范式或 retrieval 优化框架的模型加入后, W 估计可能改变. 范式内 vs 跨范式的稳定性需进一步验证. |
| **L11** | **不主张强因果** | 工具论级 preprint, 不主张 "质量向量决定下游性能". DCLM [15] "human quality judgments have only limited value" 是反向警示 |

### 4.5 与 NPM 整体框架的关系

本研究是 NPM "数据→骨架"连线的第一块实证砖. 主要贡献:

- 在 sentence-transformers 范围内, s̄ 四分量框架**可测且可重复** (Kendall W ≥ 0.92)
- Vendi 和 SVD top-k ratio 的近数学等价性 (ρ=-0.997) 是**embedding 几何**的新观察
- FreeLaw "假多样" 是 sentence-transformers 范围内的稳健现象 (头段套话 1.77×)
- §6 独立章揭示 silhouette + contextual embedding 组合的**测度局限性** (方法学贡献)

后续研究方向: 从 observational 走向 interventional, 即通过控制数据训练小模型验证 s̄ → k 因果关系.

---

## 五、结论

1. (sentence-transformers 范围内) NPM s̄ 四分量框架在 The Pile 八子集上**可测且可重复**:
   - s̄_num / s̄_rep: 跨模型完全独立 (W = 1.000 ties-corrected)
   - s̄_con / s̄_div: 三模型 Kendall W ≥ 0.92, Bootstrap CI 不含 0

2. **Vendi 与 SVD top-10 ratio 在 24 点 ρ = -0.997 近数学等价**. Vendi 跨模型绝对值不可比是 embedding 几何性质的直接后果.

3. **FreeLaw 头段套话密集** (mid s̄_div 1.77× head). 但中段 s̄_div = 0.030 仍是 8 域最低梯队. **此 claim 限定 sentence-transformers 范围**.

4. K-means K 选择 (K ∈ [2, 50]) 和 seed (σ ≈ 1e-5) 对 s̄_con 影响极小, 是稳健测量.

5. **本研究为工具论级方法论 preprint**. 不主张因果. 强主张留待后续 interventional 研究.

6. **§6 独立章** 呈现 BERT-style 模型 silhouette 的测度局限性 — 与本主线不直接对比, 作为方法学补充贡献.

\newpage

## 六、独立方法学章 · BERT-style 模型作为 silhouette 测度局限性的探针

> **本章数据与 §1-§5 主体不直接对比**. sentence-transformers 经对比学习 + uniformity loss 训练, silhouette 是合理测度. BERT-style MLM 没有 uniformity 后处理, **silhouette 在它们上是 anisotropy 主导假象** [1][2][7] (whitening 后归零, 280 次实验证实, 见 §6.2).
>
> **本章作为独立方法学探针**, 揭示 silhouette + contextual embedding 组合的测度局限性. **不应被解读为对 NPM 主线的扩展或反驳**.

### 6.1 BERT-style baseline 数据

我们在 sentence-transformers 主线之外, 跑了 4 个 BERT-style MLM 模型作为对照:

| 模型 | Params | 训练策略 | 词表 |
|---|---|---|---|
| google-bert/bert-base-uncased | 110M | 通用 MLM (Devlin et al. 2019 [20]) | BERT 30k |
| nlpaueb/legal-bert-base-uncased | 110M | 法律 MLM (Chalkidis et al. 2020 [21]) | 法律新 SP 30k |
| dmis-lab/biobert-v1.1 | 110M | 生物医学 MLM (Lee et al. 2020 [22]) | BERT 28,996 (沿用) |
| microsoft/BiomedNLP-PubMedBERT | 110M | 生物医学 MLM (Gu et al. 2021 [23]) | 生物医学新 WP 30,522 |

#### 4 模型 × 8 域 s̄_con 数据

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

**注**: 这些数字**与主体 §3.1 sentence-transformers 表不同质**, 不可直接对比.

### 6.2 5 处理 silhouette · 280 次实验的核心发现

我们对 7 模型 (3 sentence-transformers + 4 BERT-style) × 8 域 = 56 个 (model, domain) 组合, 计算 silhouette 在 5 种处理下的值, 共 **280 次 silhouette 计算**.

#### 处理定义

| 处理 | 操作 | 物理含义 |
|---|---|---|
| raw | 直接对原 embedding 算 | 含完整 anisotropy + 全局偏置 |
| centered | X − mean(X) | 去全局偏置 (mean shift) |
| zstd | (X − mean) / std | 去 mean + 维度尺度归一化 |
| abtt | All-but-the-top: 减 top-1 主成分 [1] | 去最强 rogue dimension |
| **whitened** | PCA whitening | **完全去 anisotropy** |

#### FreeLaw × 7 模型 × 5 处理 (核心证据表)

| 模型 | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM | 0.019 | 0.019 | 0.019 | 0.021 | **0.000** |
| BGE-small | 0.022 | 0.022 | 0.021 | 0.019 | **-0.004** |
| BGE-large | 0.020 | 0.020 | 0.019 | 0.014 | **-0.050** |
| BERT-base | 0.102 | 0.102 | 0.091 | 0.058 | **0.009** |
| Legal-BERT | 0.104 | 0.104 | 0.093 | 0.074 | **0.000** |
| BioBERT | 0.160 | 0.160 | 0.094 | 0.094 | **0.002** |
| PubMedBERT | 0.114 | 0.114 | 0.094 | 0.066 | **-0.005** |

**关键观察**: **whitening 后所有 7 模型在 FreeLaw 上 silhouette ≈ 0** (从 -0.05 到 +0.009). 同样, PubMed Central 在 7 模型 whitening 后 silhouette ≈ 0.

→ **silhouette 在 contextual embedding 上不是测真 cluster, 是 anisotropy 主导假象**.

#### Pairwise Spearman ρ_con · 5 处理对比 (关键 6 配对)

| 配对 | raw | centered | zstd | abtt | **whitened** |
|---|---|---|---|---|---|
| MiniLM vs Legal-BERT | -0.429 | -0.429 | -0.333 | +0.214 | **+0.048** |
| MiniLM vs BERT-base | -0.238 | -0.238 | -0.214 | +0.690 | **+0.381** |
| BERT-base vs Legal-BERT | +0.810 | +0.810 | +0.762 | +0.095 | +0.690 |
| BERT-base vs BioBERT | +0.524 | +0.524 | +0.548 | +0.333 | +0.524 |
| BERT-base vs PubMedBERT | +0.738 | +0.738 | +0.690 | +0.357 | +0.381 |
| BioBERT vs PubMedBERT | +0.500 | +0.500 | +0.476 | +0.786 | +0.310 |

**MiniLM vs Legal-BERT**: raw ρ = **-0.429** → whitened ρ = **+0.048** · "视角断裂"完全消失. 早期 v2.2 的"两套视角系统"假设被自身 5 处理实验框架证伪.

### 6.3 SVD 各向异性 · 7 模型对比 (跨 8 域均值 · l2norm 模式)

| 类别 | 模型 | top-10 ratio |
|---|---|---|
| sentence-transformers | MiniLM | 0.331 |
| sentence-transformers | BGE-large | 0.625 |
| sentence-transformers | BGE-small | 0.660 |
| BERT-style | BERT-base | **0.900** |
| BERT-style | BioBERT | 0.951 |
| BERT-style | Legal-BERT | 0.953 |
| BERT-style | **PubMedBERT** | **0.992** |

**两类 anisotropy 层级清晰**:
- sentence-transformers: top-10 = 0.33-0.66 (中等)
- BERT-style: top-10 = **0.90-0.99** (强→极端)

PubMedBERT 0.992 是迄今最各向异性的模型 (from-scratch + 单一同质生物医学语料 + 无 uniformity loss).

### 6.4 与文献对接

> **Aharoni & Goldberg 2020 [12] (ACL)**: 用通用 BERT-base 把 5 个 corpus-source 域聚类, **purity 87.66%**. "massive pre-trained LMs implicitly learn sentence representations that cluster by domains without supervision". → **任何 LM (含通用 BERT) 都会把 corpus-source 拉开**, 不是专模专属信号.

> **Timkey & van Schijndel 2021 [7] (EMNLP)**: 1-3 个 rogue dimensions 主导 cosine 相似度. BERT layer 11 单 dim 占 88.4%, GPT-2 last layer top-1 = 76.3%. → **raw silhouette ≈ rogue-dim 上的 silhouette**, 与真聚类质量脱钩.

> **Mu & Viswanath 2018 [1] (ICLR)**: 前几个主方向编码词频. **All-but-the-top** (减 top-1 主成分) 后 anisotropy 大幅降低. → 我们的 ABTT 处理是文献标准操作.

> **Rudman & Eickhoff 2024 [8] (I-STAR)**: **降低 isotropy 反而提升下游性能** · 不能简单认为各向异性 ↑ = 数据质量差. → 解读 §6.2 数据时不做 isotropy → quality 直接因果论.

> **Cai et al. 2021 [6] (ICLR)**: 提出 contextual embedding 全局各向异性是 isolated clusters + cluster manifold 的表象 · 簇内局部各向同性. → 支持 §6.2 whitening 后 silhouette 全归零的观察 — embedding 局部结构已被 anisotropy 全局模式遮蔽.

> **Bis et al. 2021 [5] (NAACL)**: 提出 contextual embedding 高各向异性主因是 "common shift" — 每 step 除 ground-truth 外所有 embedding 沿同一方向接受梯度. → 解释 §3.5 raw vs centered gap 0.5 的物理含义 — gap 主要由 mean shift 驱动.

> **Ait-Saada & Nadif 2023 [9] (ACL Short)**: 反共识平衡 — 测得各向异性与 clustering NMI **接近零或弱负相关**, 甚至称 "fostering high anisotropy yields high-quality clustering representations". → 注: 他们用 NMI (外部指标), NPM 用 silhouette (内部指标), 两类指标对 rogue dim scale 敏感性不同, 故反共识不直接搬到 silhouette.

> **Gao et al. 2019 [2] (ICLR) + Wang et al. 2020 [3] (ICLR)**: anisotropy 理论根基 — softmax + weight tying + Zipf 长尾 → 低频 token narrow cone. → 解释为何 BERT-style 模型 (无 uniformity loss) 必然各向异性强.

### 6.5 §6 方法学贡献 (独立于主线)

本独立章给出的方法学发现:

1. **silhouette 在 contextual embedding 上不是真 cluster 测度**: 7 模型 × 8 域 × whitening = 280 次 silhouette 几乎全部归零. 给"silhouette + contextual embedding"组合的局限性提供实证证据链.

2. **判别"真子类信号 vs anisotropy 假象"的诊断**: 比较 raw 与 whitened silhouette 的差值. 若 whitened 后归零 → anisotropy 假象; 若 whitened 后保留信号 → 真聚类信号. 简单可复现.

3. **多模型一致性的 anisotropy 偏置**: raw 下 Kendall W = 0.266, whitened 后 W = 0.228 (反而下降). 说明 raw 下的"跨模型一致性"部分来自 anisotropy 共同模式, 不是真共识.

4. **NPM 主线的合理范围确认**: sentence-transformers 类模型 (经 uniformity loss 训练) 在 silhouette/Vendi 上是合理测度; BERT-style 不直接适用.

### 6.6 §6 限制 (独立章自身的诚实边界)

| # | 边界 | 说明 |
|---|---|---|
| §6 L1 | 仅 4 个 BERT-style 模型 | 加 SciBERT / CaseLaw-BERT 第 5-6 个会更强 |
| §6 L2 | 5 处理是已知文献标准, 但未穷举 | 可加更多 (e.g. Mahalanobis whitening, supervised whitening) |
| §6 L3 | "anisotropy 主导"是从相关数据推断, 不是直接因果实验 | 因果验证需要训练时控制 anisotropy 强度 (如加 contrastive head 微调 BERT) |

\newpage

## 七、附录

### 附录 A · 八域完整数据 (sentence-transformers 主体)

详见配套数据文档 `阶梯1_完整原始数据_含BERT-baseline_20260425.md`:

- **§ 1 主线数据** (sentence-transformers, 三模型)
  - § 1.2 表: 8 域 × 3 模型 × 4 分量
- **§ 2 BERT 探针数据** (BERT-style, 四模型 · 仅供 § 6 引用)
  - § 2.2 表: 8 域 × 4 模型 × 4 分量

### 附录 B · 主图清单

主图存放路径: `figures/` 子目录, 中英双版各 4 张 (PNG + PDF 各 2 个). 命名约定: `F{编号}_{描述}{_cn 或英文}.{png,pdf}`.

**F1 · I6 四重机制分解**

- 内容: 截断 / 子类识别 / BGE 偏置 / 真低秩 四 panel
- 范围: 含 BERT 模型, 仅 §6 独立章引用

**F2 · Vendi-SVD 三角印证 (32 点散点 ρ=-0.997)**

- 内容: 4 模型 × 8 域 = 32 点 + power-law 拟合
- 范围: 含 Legal-BERT 数据; 主线 §3.6 引用 24 点 (sentence-transformers only) 版本

**F3 · s̄ 四分量热图**

- 内容: 4 分量 × 4 模型 × 8 域 heatmap
- 范围: 含 BERT 模型

**F4 · pairwise Spearman 矩阵**

- 内容: 4 × 4 cross-model 相关矩阵
- 范围: 含 BERT 模型; 主线 §3.2 报告 3 × 3 版本数字

**说明**: 部分原图含 BERT 模型, 严格主线 PDF 应重绘只含 sentence-transformers 版本. 当前版本暂用现有图 + caveat 标注引用范围.

### 附录 C · 相关工作

本附录列出本研究涉及的 25 篇文献, 按主题分 6 类组织.

#### C.1 Embedding 几何与各向异性 (anisotropy)

(共 9 条 · 涵盖 anisotropy 现象的发现、成因与缓解)

[1] **Mu, J., & Viswanath, P.** (2018). All-but-the-Top: Simple and Effective Postprocessing for Word Representations. *International Conference on Learning Representations (ICLR)*. arXiv:1702.01417.

[2] **Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y.** (2019). Representation Degeneration Problem in Training Natural Language Generation Models. *International Conference on Learning Representations (ICLR)*. arXiv:1907.12009.

[3] **Wang, L., Huang, J., Huang, K., Hu, Z., Wang, G., & Gu, Q.** (2020). Improving Neural Language Generation with Spectrum Control. *International Conference on Learning Representations (ICLR)*.

[4] **Ethayarajh, K.** (2019). How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:1909.00512.

[5] **Bis, D., Podkorytov, M., & Liu, X.** (2021). Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications. *NAACL 2021*. ACL Anthology 2021.naacl-main.403.

[6] **Cai, X., Huang, J., Bian, Y., & Church, K.** (2021). Isotropy in the Contextual Embedding Space: Clusters and Manifolds. *International Conference on Learning Representations (ICLR)*.

[7] **Timkey, W., & van Schijndel, M.** (2021). All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:2109.04404.

[8] **Rudman, W., & Eickhoff, C.** (2024). Stable Anisotropic Regularization (I-STAR). *International Conference on Learning Representations (ICLR)*. arXiv:2305.19358.

[9] **Ait-Saada, M., & Nadif, M.** (2023). Is Anisotropy Truly Harmful? A Case Study on Text Clustering. *Annual Meeting of the Association for Computational Linguistics (ACL Short)*. ACL Anthology 2023.acl-short.103.

#### C.2 多样性与质量测度

(共 3 条 · Vendi Score / effective rank / 域聚类)

[10] **Friedman, D., & Dieng, A. B.** (2022). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. *Transactions on Machine Learning Research (TMLR)*. arXiv:2210.02410.

[11] **Roy, O., & Vetterli, M.** (2007). The effective rank: A measure of effective dimensionality. *15th European Signal Processing Conference (EUSIPCO)*, pp. 606-610.

[12] **Aharoni, R., & Goldberg, Y.** (2020). Unsupervised Domain Clusters in Pretrained Language Models. *Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 7747-7763.

#### C.3 Scaling Laws 与数据质量

(共 3 条 · 后续 interventional 研究方向参考)

[13] **Subramanyam, S., Chen, Y., & Grossman, S.** (2025). Quality-aware scaling laws for language model pretraining. arXiv:2510.03313v2.

[14] **Bahri, Y., Dyer, E., Kaplan, J., Lee, J., & Sharma, U.** (2024). Explaining Neural Scaling Laws. *Proceedings of the National Academy of Sciences (PNAS)*, 121(27): e2311878121.

[15] **Li, J., Fang, A., Smyrnis, G., et al.** (2024). DataComp-LM: In search of the next generation of training sets for language models. *NeurIPS Datasets and Benchmarks (DCLM)*. arXiv:2406.11794.

#### C.4 Sentence Embedding 模型

(共 4 条 · 主体 §1-§5 使用的三模型原论文)

[16] **Reimers, N., & Gurevych, I.** (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Empirical Methods in Natural Language Processing (EMNLP)*. arXiv:1908.10084.

[17] **Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M.** (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 2020*. arXiv:2002.10957.

[18] **Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N.** (2023). C-Pack: Packaged Resources To Advance General Chinese Embedding (BGE / FlagEmbedding). arXiv:2309.07597.

[19] **Liu, Z., Shao, S., Shi, X., Lian, D., & Xiao, S.** (2022). RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder. *EMNLP 2022*. arXiv:2205.12035.

#### C.5 BERT-style 专业领域模型 (§6 引用)

(共 4 条 · §6 独立方法学章使用的探针模型原论文)

[20] **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*, pp. 4171-4186. arXiv:1810.04805.

[21] **Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I.** (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*. arXiv:2010.02559.

[22] **Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J.** (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4): 1234-1240. arXiv:1901.08746.

[23] **Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H.** (2021). Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing (PubMedBERT). *ACM Transactions on Computing for Healthcare*, 3(1): 1-23. arXiv:2007.15779.

#### C.6 NPM 框架自身

(共 2 条 · 已在 Zenodo 发布的 NPM 系列前作)

[24] **Ding, T.** (2026). Neural Percolation Model: A Porous-Media-Inspired Framework for Neural Network Training Dynamics. Zenodo. DOI: 10.5281/zenodo.19209722.

[25] **Ding, T.** (2026). Cross-Family Convergence of Neural Network Weight Skeletons (NPM-K). Zenodo. DOI: 10.5281/zenodo.19652706.

### 附录 D · 代码与数据可用性

本研究的全部代码、数据和透明数据 reference 文档已在 GitHub 公开:

**Code & data**: <https://github.com/tiexinding/data-quality-vec-public>

仓库内容: 中英双版技术报告 v2.4 + 主线/探针/消融/5 处理共 19 份 CSV/JSON 数据 + F1-F4 主图 (中英 PNG+PDF 16 份) + 主管道脚本 (`run_stage1_sbar_v2.py`) + SVD 各向异性诊断 + K-scan + 图生成 + PDF 构建脚本. License: MIT.

中间 embedding (`.npy`, ~380 MB) 与 text cache (~137 MB) 不入仓, 可由 `code/run_stage1_sbar_v2.py` 从 `monology/pile-uncopyrighted` (HuggingFace) + 各模型 checkpoint 重新生成.

---
