#!/bin/bash
# ==============================================================================
# Evaluation: Tree Mode (EAGLE-style)
# ==============================================================================

set -e

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"

# Target & Decode
TARGET_MODE="fp"
DECODE_MODE="tree"
TREE_PRESET="default"
TOP_K=10

# Benchmark
BENCHMARK="mt_bench"
MAX_SAMPLES=20
MAX_NEW_TOKENS=128
DRAFT_LENGTHS="1,3,5,7"
MODE="greedy"

OUTPUT_FILE="eval_results/speculative_tree_eval.json"
DEVICE="cuda"

# ===========================
# VALIDATE
# ===========================

if [[ "${DRAFT_MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
    echo "ERROR: Set DRAFT_MODEL_PATH to actual trained model directory."
    exit 1
fi

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Evaluation: Tree Speculative Decoding"
echo "  Draft:       ${DRAFT_MODEL_PATH} (0.1-bit)"
echo "  Target:      ${MODEL_ID} (${TARGET_MODE})"
echo "  Tree Preset: ${TREE_PRESET}"
echo "  Benchmark:   ${BENCHMARK}"
echo "============================================================"

mkdir -p $(dirname ${OUTPUT_FILE})

python eval_speculative.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode ${TARGET_MODE} \
    --decode_mode ${DECODE_MODE} \
    --tree_preset ${TREE_PRESET} \
    --top_k ${TOP_K} \
    --benchmark ${BENCHMARK} \
    --max_samples ${MAX_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_lengths ${DRAFT_LENGTHS} \
    --mode ${MODE} \
    --output_file ${OUTPUT_FILE} \
    --device ${DEVICE}

echo "Done! Results: ${OUTPUT_FILE}"
