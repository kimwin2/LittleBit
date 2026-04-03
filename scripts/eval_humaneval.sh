#!/bin/bash
# ==============================================================================
# HumanEval Evaluation: Accuracy (pass@1) + Speculative Decoding Speedup
# ==============================================================================
#
# Evaluates:
#   1. FP16 baseline model            → pass@1
#   2. Draft model (0.3-bit)          → pass@1
#   3. Target model (0.3+1.7 = 2bit)  → pass@1
#   4. Speculative decoding            → pass@1 + tokens/s + speedup
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Draft model (0.3-bit, Step 1 output)
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01"

# Residual model (1.7-bit, Step 2 output)
RESIDUAL_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01"

# Evaluation settings
MAX_PROBLEMS=164          # All 164 HumanEval problems
MAX_NEW_TOKENS=512        # Max tokens per generation
EXEC_TIMEOUT=5            # Code execution timeout (seconds)
DRAFT_LENGTHS="1,3,5,7"   # Draft lengths for speculative decoding
OUTPUT_DIR="eval_results/humaneval"

# What to evaluate (set to empty string "" to skip)
EVAL_FP="--eval_fp"                     # FP16 baseline
EVAL_DRAFT="--eval_draft"               # Draft model
EVAL_TARGET="--eval_target"             # Target (draft+residual)
EVAL_SPECULATIVE="--eval_speculative"   # Speculative decoding speedup

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

if [ ! -d "${RESIDUAL_MODEL_PATH}" ]; then
    echo "ERROR: Residual model not found: ${RESIDUAL_MODEL_PATH}"
    exit 1
fi

echo "============================================================"
echo "HumanEval Evaluation"
echo "  Base model:  ${BASE_MODEL_ID}"
echo "  Draft:       ${DRAFT_MODEL_PATH}"
echo "  Residual:    ${RESIDUAL_MODEL_PATH}"
echo "  Problems:    ${MAX_PROBLEMS}"
echo "  Output:      ${OUTPUT_DIR}"
echo "============================================================"

# ===========================
# RUN EVALUATION
# ===========================

python eval_humaneval.py \
    --base_model_id "${BASE_MODEL_ID}" \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --residual_model_path "${RESIDUAL_MODEL_PATH}" \
    ${EVAL_FP} \
    ${EVAL_DRAFT} \
    ${EVAL_TARGET} \
    ${EVAL_SPECULATIVE} \
    --max_problems ${MAX_PROBLEMS} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --exec_timeout ${EXEC_TIMEOUT} \
    --draft_lengths "${DRAFT_LENGTHS}" \
    --output_dir "${OUTPUT_DIR}"
