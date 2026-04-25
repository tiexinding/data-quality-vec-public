#!/bin/bash
# run_all_ablation.sh · 一夜挂机 · P0-1b/1c + P0-2 + P0-3 + P0-1d 全串起
# B2 · 2026-04-24
set -e  # 任何命令失败立即退出; tee 保证 log 完整

BASE=/root/autodl-tmp
cd "$BASE"

export HF_HOME=$BASE/hf_cache
export HF_DATASETS_CACHE=$BASE/hf_cache/datasets
export HF_ENDPOINT=https://hf-mirror.com

PY=/root/miniconda3/bin/python
SUBSETS='Pile-CC Wikipedia (en) ArXiv Github PubMed Central FreeLaw StackExchange USPTO Backgrounds'
# Bash nightmare: space in "Wikipedia (en)" / "PubMed Central" / "USPTO Backgrounds"
# We pass them via an env array.
read -ra SUBSETS_ARR <<<"Pile-CC|Wikipedia (en)|ArXiv|Github|PubMed Central|FreeLaw|StackExchange|USPTO Backgrounds"
# Use | as separator → split into array
IFS='|' read -ra SUBSETS_ARR <<<"Pile-CC|Wikipedia (en)|ArXiv|Github|PubMed Central|FreeLaw|StackExchange|USPTO Backgrounds"

echo "=========================================="
echo " ABLATION PIPELINE · $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# ---------------- Step 0 : MiniLM rerun with save_embeddings (for K-scan + SVD) ----------------
# NOTE: sbar results will match v1 (uses same model+pos), but we get the .npy files we need.
echo ">>> STEP 0 · MiniLM + save_embeddings (8 domains, head)"
$PY -u run_stage1_sbar_v2.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --n_samples 5000 \
    --output_dir results_minilm \
    --cache_dir cache_minilm \
    --emb_dir results_minilm \
    --save_embeddings \
    --truncation_pos head \
    --subsets "${SUBSETS_ARR[@]}" \
    2>&1 | tee logs/step0_minilm_head.log

# ---------------- Step 1 : BGE-large ----------------
echo ">>> STEP 1 · BGE-large + save_embeddings (8 domains, head)"
$PY -u run_stage1_sbar_v2.py \
    --model_name BAAI/bge-large-en-v1.5 \
    --n_samples 5000 \
    --output_dir results_bge_large \
    --cache_dir cache_bge_large \
    --emb_dir results_bge_large \
    --save_embeddings \
    --truncation_pos head \
    --subsets "${SUBSETS_ARR[@]}" \
    2>&1 | tee logs/step1_bge_large_head.log

# ---------------- Step 2 : Legal-BERT with mean-pool wrapper ----------------
echo ">>> STEP 2 · Legal-BERT + save_embeddings (8 domains, head)"
$PY -u run_stage1_sbar_v2.py \
    --model_name nlpaueb/legal-bert-base-uncased \
    --use_wrapper \
    --n_samples 5000 \
    --output_dir results_legal_bert \
    --cache_dir cache_legal_bert \
    --emb_dir results_legal_bert \
    --save_embeddings \
    --truncation_pos head \
    --subsets "${SUBSETS_ARR[@]}" \
    2>&1 | tee logs/step2_legal_bert_head.log

# ---------------- Step 3 : BGE-small rerun with save_embeddings (补回 emb .npy) ----------------
echo ">>> STEP 3 · BGE-small rerun + save_embeddings (for SVD)"
$PY -u run_stage1_sbar_v2.py \
    --model_name BAAI/bge-small-en-v1.5 \
    --n_samples 5000 \
    --output_dir results_bge_small \
    --cache_dir cache_bge_small \
    --emb_dir results_bge_small \
    --save_embeddings \
    --truncation_pos head \
    --subsets "${SUBSETS_ARR[@]}" \
    2>&1 | tee logs/step3_bge_small_head.log

# ---------------- Step 4 : P0-3 Truncation · MiniLM × FreeLaw+ArXiv × {mid, tail} ----------------
echo ">>> STEP 4a · P0-3 trunc: MiniLM FreeLaw+ArXiv MID"
$PY -u run_stage1_sbar_v2.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --n_samples 5000 \
    --output_dir results_trunc_mid \
    --cache_dir cache_minilm \
    --emb_dir results_trunc_mid \
    --save_embeddings \
    --truncation_pos mid \
    --subsets FreeLaw ArXiv \
    2>&1 | tee logs/step4a_minilm_mid.log

echo ">>> STEP 4b · P0-3 trunc: MiniLM FreeLaw+ArXiv TAIL"
$PY -u run_stage1_sbar_v2.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --n_samples 5000 \
    --output_dir results_trunc_tail \
    --cache_dir cache_minilm \
    --emb_dir results_trunc_tail \
    --save_embeddings \
    --truncation_pos tail \
    --subsets FreeLaw ArXiv \
    2>&1 | tee logs/step4b_minilm_tail.log

# ---------------- Step 5 : P0-2 K scan + seed scan on MiniLM ----------------
echo ">>> STEP 5 · K+seed scan on MiniLM embeddings"
$PY -u run_p0_2_kscan.py \
    --emb_dir results_minilm \
    --output_dir results_kscan \
    --domains FreeLaw StackExchange PubMed_Central \
    --pos head \
    2>&1 | tee logs/step5_kscan.log

# ---------------- Step 6 : P0-1d Anisotropy SVD across 4 models ----------------
echo ">>> STEP 6 · Anisotropy SVD · 4 models × 8 domains"
$PY -u run_p0_1d_anisotropy.py \
    --emb_dirs minilm:results_minilm bge_small:results_bge_small bge_large:results_bge_large legal_bert:results_legal_bert \
    --output_dir anisotropy_out \
    --pos head \
    2>&1 | tee logs/step6_anisotropy.log

# ---------------- Final: tarball and done ----------------
STAMP=$(date +%Y%m%d_%H%M)
tar -czf "ablation_all_${STAMP}.tar.gz" \
    results_minilm results_bge_small results_bge_large results_legal_bert \
    results_trunc_mid results_trunc_tail results_kscan anisotropy_out \
    cache_minilm cache_bge_small cache_bge_large cache_legal_bert \
    logs/ \
    2>/dev/null || true

echo "=========================================="
echo " ABLATION DONE · $(date '+%Y-%m-%d %H:%M:%S')"
echo " Tarball: ablation_all_${STAMP}.tar.gz"
echo "=========================================="
