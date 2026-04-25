# Data Quality Vectorization Framework

A measurability and cross-model stability study on a four-component data quality vector (s̄_con / s̄_num / s̄_rep / s̄_div) over eight subsets of The Pile, with a standalone methodological probe on BERT-style models.

This repository accompanies the v2.4 technical report (bilingual: English + Chinese) and contains all code, numerical results, figures, and a transparent data reference.

## What this is

A tool-paper-level methodology preprint. Concretely:

- **Main study (paper §1-§5)**: We test whether the s̄ four-component framework is **measurable** and **cross-model robust** within the sentence-transformers class (MiniLM / BGE-small / BGE-large), across 8 Pile subsets.
- **Standalone probe (paper §6)**: We test the same metrics on four BERT-style MLM models (BERT-base / Legal-BERT / BioBERT / PubMedBERT) under five treatments (raw / centered / zstd / abtt / whitened), totaling 280 silhouette computations. The result reveals that silhouette on BERT-style contextual embeddings is anisotropy-dominated, not a real cluster signal.

The two data classes are **NOT directly compared** because the two model classes have fundamentally different training objectives. See the paper §1.4 Scope Declaration and `docs/Full_Raw_Data_with_BERT_baseline_EN.md` §0 for the rules of engagement.

This framework originates from the data-vectorization tenet of the Neural Percolation Model (NPM) but stands as an independent measurement framework — it does not require the reader to accept the broader physical picture of NPM.

## Key findings (within sentence-transformers scope)

1. **Kendall's W ≥ 0.88** across the three models on all four components; s̄_num / s̄_rep are theoretically identical (W = 1.000 ties-corrected); s̄_con W = 0.878, s̄_div W = 0.915.
2. **Vendi and SVD top-10 ratio are near-mathematically equivalent**: 24 points (3 models × 8 domains) Spearman ρ = **-0.997**, with a power-law fit s̄_div ∝ x^(-4.5).
3. **FreeLaw head boilerplate concentration**: head→mid s̄_div ↑ **1.77×** in slice experiment; mid section is more diverse than head, independent of embedding choice.
4. **FreeLaw is in the lowest-diversity tier** under all three sentence-transformers models (rank 1-2 lowest).

## Repository layout

```
data-quality-vec-public/
├── README.md                                       # This file
├── LICENSE                                         # MIT
├── requirements.txt                                # Python dependencies
├── papers/
│   ├── Stage1_Technical_Report_EN_v2.4.{md,pdf}    # English technical report
│   └── Stage1_Technical_Report_CN_v2.4.{md,pdf}    # Chinese technical report
├── code/
│   ├── run_stage1_sbar_v2.py                       # Main s̄ four-component pipeline
│   ├── run_p0_1d_anisotropy.py                     # SVD anisotropy diagnostic
│   ├── run_p0_2_kscan.py                           # K-scan ablation
│   ├── make_all_figures.py                         # Figure generation (F1-F4)
│   ├── build_v24_pdfs.py                           # Pandoc + xelatex PDF builder
│   ├── run_all_ablation.sh                         # End-to-end ablation driver
│   ├── smoke_v2.py                                 # Smoke test
│   └── ...
├── data/
│   ├── primary/                                    # sentence-transformers (main study)
│   ├── probe/                                      # BERT-style (§6 probe)
│   ├── ablation/                                   # K-scan, slice, anisotropy
│   └── treatments_5/                               # 280-row 5-treatment silhouette
├── figures/                                        # F1-F4 (PNG + PDF, EN + CN)
└── docs/
    ├── Full_Raw_Data_with_BERT_baseline_EN.md      # Transparent data reference (English)
    └── Full_Raw_Data_with_BERT_baseline_CN.md      # Same content, Chinese
```

## Reproducing

### Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `sentence-transformers`, `transformers`, `torch`, `scikit-learn`, `datasketch` (MinHash), `vendi-score`, `scipy`, `pandas`, `matplotlib`.

### Main pipeline

```bash
# Compute s̄ four components for one model on the 8 Pile subsets
python code/run_stage1_sbar_v2.py --model all-MiniLM-L6-v2 --output_dir data/primary/

# SVD anisotropy diagnostic (across 7 models)
python code/run_p0_1d_anisotropy.py --output_dir data/ablation/

# K-scan ablation
python code/run_p0_2_kscan.py --output_dir data/ablation/

# Regenerate figures (uses CSVs in data/)
python code/make_all_figures.py

# Build PDFs from the markdown reports
python code/build_v24_pdfs.py
```

> Note: Intermediate embeddings (~380 MB total) and text caches (~137 MB) are not included in this repository. They will be regenerated on the fly from `monology/pile-uncopyrighted` (HuggingFace) and the listed model checkpoints. Expect the first run to take significant disk and time.

### Cross-checking numbers

Every number in the v2.4 paper is traceable to a CSV/JSON in `data/`. The transparent reference document `docs/Full_Raw_Data_with_BERT_baseline_EN.md` lists exact tables — you should be able to point at any sentence in the paper and find the underlying number in this repository within seconds.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19762059.svg)](https://doi.org/10.5281/zenodo.19762059)

**Zenodo DOI**: [10.5281/zenodo.19762059](https://doi.org/10.5281/zenodo.19762059)

```bibtex
@misc{ding2026dataqualityvec,
  author       = {Ding, Tiexin},
  title        = {A Data Quality Vectorization Framework for Neural Networks
                  · Measurability and Cross-Model Stability Study},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2.4},
  doi          = {10.5281/zenodo.19762059},
  url          = {https://doi.org/10.5281/zenodo.19762059},
  howpublished = {\url{https://github.com/tiexinding/data-quality-vec-public}}
}
```

## Related work

- **Neural Percolation Model (NPM)**: parent framework. Zenodo DOI [10.5281/zenodo.19209722](https://doi.org/10.5281/zenodo.19209722).
- **Cross-Family Convergence of Neural Network Weight Skeletons (NPM-K)**: Zenodo DOI [10.5281/zenodo.19652706](https://doi.org/10.5281/zenodo.19652706); GitHub [tiexinding/NPM-K-public](https://github.com/tiexinding/NPM-K-public).

## Contact

Tiexin Ding · Independent Research · `tiexinding [at] gmail [dot] com`

## License

MIT — see `LICENSE`.
