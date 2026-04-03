#!/bin/bash
# ==============================================================================
# HumanEval Accuracy Only (no speculative decoding)
# ==============================================================================
#
# Quick pass@1 comparison: FP16 vs Draft vs Target
# Skips speculative decoding speedup measurement for faster evaluation.
#
# ==============================================================================

set -e

BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01"
RESIDUAL_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01"

echo "============================================================"
echo "HumanEval Accuracy: FP16 vs Draft vs Target (pass@1)"
echo "============================================================"

python eval_humaneval.py \
    --base_model_id "${BASE_MODEL_ID}" \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --residual_model_path "${RESIDUAL_MODEL_PATH}" \
    --eval_fp \
    --eval_draft \
    --eval_target \
    --max_problems 164 \
    --max_new_tokens 512 \
    --output_dir "eval_results/humaneval_accuracy"
