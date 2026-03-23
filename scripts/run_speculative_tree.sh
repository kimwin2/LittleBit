#!/bin/bash
# ==============================================================================
# Speculative Decoding: Tree Mode (EAGLE-style)
# ==============================================================================
#
# Draft:  0.1-bit quantized model (tree candidate generation)
# Target: FP or Matryoshka model (parallel tree verification)
#
# Tree mode generates a tree of candidates and verifies all branches
# in ONE target model forward pass, improving acceptance rate.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# >>> Set to your trained Step 1 draft model <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"

# Target mode: "fp" (pre-Step2) or "matryoshka" (post-Step2)
TARGET_MODE="fp"
# RESIDUAL_MODEL_PATH=  # only needed for matryoshka mode

# Decode mode
DECODE_MODE="tree"
TREE_PRESET="default"     # "small", "default", "large"
TOP_K=10

# Generation
PROMPT="Write a Python function to compute fibonacci numbers efficiently using memoization."
MAX_NEW_TOKENS=256
MODE="greedy"
TEMPERATURE=1.0
COMPARE_BASELINE="true"
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
echo "Speculative Decoding (Tree Mode - EAGLE-style)"
echo "  Draft:       ${DRAFT_MODEL_PATH} (0.1-bit)"
echo "  Target:      ${MODEL_ID} (${TARGET_MODE})"
echo "  Tree Preset: ${TREE_PRESET}"
echo "  Top-K:       ${TOP_K}"
echo "  Mode:        ${MODE}"
echo "============================================================"

python speculative_decoding.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --target_mode ${TARGET_MODE} \
    --decode_mode ${DECODE_MODE} \
    --tree_preset ${TREE_PRESET} \
    --top_k ${TOP_K} \
    --prompt "${PROMPT}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --mode ${MODE} \
    --temperature ${TEMPERATURE} \
    --compare_baseline ${COMPARE_BASELINE} \
    --device ${DEVICE}
