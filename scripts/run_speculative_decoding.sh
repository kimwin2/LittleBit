#!/bin/bash
# ==============================================================================
# Speculative Decoding: Interactive Generation
# ==============================================================================
#
# Run speculative decoding with the Matryoshka draft (0.1-bit) and
# target (0.1-bit + 0.9-bit) models.
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# >>> Set these to your trained model directories <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"
RESIDUAL_MODEL_PATH="outputs/step2_residual_0.9bit/REPLACE_WITH_TIMESTAMP"

# Generation
PROMPT="Write a Python function to compute fibonacci numbers efficiently using memoization."
MAX_NEW_TOKENS=256
DRAFT_LENGTH=5
MODE="greedy"        # "greedy" or "sampling"
TEMPERATURE=1.0
COMPARE_BASELINE="true"

# Device
DEVICE="cuda"

# ===========================
# VALIDATE
# ===========================

for MODEL_PATH in "${DRAFT_MODEL_PATH}" "${RESIDUAL_MODEL_PATH}"; do
    if [[ "${MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
        echo "ERROR: Please set model paths to actual trained model directories."
        exit 1
    fi
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: Model directory not found: ${MODEL_PATH}"
        exit 1
    fi
done

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "Speculative Decoding"
echo "  Draft Model:    ${DRAFT_MODEL_PATH}"
echo "  Residual Model: ${RESIDUAL_MODEL_PATH}"
echo "  Draft Length K:  ${DRAFT_LENGTH}"
echo "  Mode:           ${MODE}"
echo "============================================================"

python speculative_decoding.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --residual_model_path ${RESIDUAL_MODEL_PATH} \
    --prompt "${PROMPT}" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_length ${DRAFT_LENGTH} \
    --mode ${MODE} \
    --temperature ${TEMPERATURE} \
    --compare_baseline ${COMPARE_BASELINE} \
    --device ${DEVICE}
