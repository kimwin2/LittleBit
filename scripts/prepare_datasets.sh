#!/bin/bash
# ==============================================================================
# Dataset Preparation: Download & Process wikitext2 + ShareGPT
# ==============================================================================
#
# Downloads, tokenizes, and saves datasets for QAT training.
# Run this ONCE before training Step 1 and Step 2.
#
# The processed datasets are cached to disk and auto-detected by training scripts.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Tokenizer model (must match the model used for training)
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# Where to save processed datasets
# Training scripts use --data_root to find these
OUTPUT_DIR="./"

# (Optional) Path to local ShareGPT .jsonl file
# If not set, will download from HuggingFace automatically
SHAREGPT_PATH=""
# Example: SHAREGPT_PATH="/data/sharegpt_train.jsonl"

# Token sequence length
BLOCK_SIZE=2048

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Dataset Preparation"
echo "  Model:       ${MODEL_ID}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Block size:  ${BLOCK_SIZE}"
echo "============================================================"

SHAREGPT_ARG=""
if [ -n "${SHAREGPT_PATH}" ]; then
    SHAREGPT_ARG="--sharegpt_path ${SHAREGPT_PATH}"
fi

python prepare_datasets.py \
    --model_id ${MODEL_ID} \
    --output_dir ${OUTPUT_DIR} \
    ${SHAREGPT_ARG} \
    --block_size ${BLOCK_SIZE}

echo "============================================================"
echo "Dataset preparation complete!"
echo "You can now run: scripts/train_step1_draft.sh"
echo "============================================================"
