#!/bin/bash
# ==============================================================================
# Speculative Decoding: Full Evaluation Suite
# ==============================================================================
#
# Run comprehensive benchmarks for speculative decoding:
# - MT-Bench (instruction following)
# - GSM8K (math reasoning)
# - HumanEval (code generation)
# - CNN/DailyMail (summarization)
#
# Reports:
# - Mean acceptance length (α)
# - Token acceptance rate
# - Speedup vs autoregressive baseline
# - Tokens per second
# - Output match rate (greedy)
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# >>> Set these to your trained model directories <<<
DRAFT_MODEL_PATH="outputs/step1_draft_0.1bit/REPLACE_WITH_TIMESTAMP"
RESIDUAL_MODEL_PATH="outputs/step2_residual_0.9bit/REPLACE_WITH_TIMESTAMP"

# Benchmark settings
BENCHMARK="all"      # "all", "mt_bench", "gsm8k", "humaneval", "summarization"
MAX_SAMPLES=50       # Samples per benchmark
MAX_NEW_TOKENS=128
DRAFT_LENGTHS="1,3,5,7"
MODE="greedy"        # "greedy" or "sampling"

# Output
OUTPUT_FILE="eval_results/speculative_eval.json"

# Device
DEVICE="cuda"

# ===========================
# VALIDATE
# ===========================

for MODEL_PATH in "${DRAFT_MODEL_PATH}" "${RESIDUAL_MODEL_PATH}"; do
    if [[ "${MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
        echo "ERROR: Please set model paths to actual trained model directories."
        echo "  Edit this script and replace REPLACE_WITH_TIMESTAMP with actual paths."
        exit 1
    fi
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: Model directory not found: ${MODEL_PATH}"
        exit 1
    fi
done

# ===========================
# RUN EVALUATION
# ===========================

echo "============================================================"
echo "Speculative Decoding Evaluation"
echo "  Draft Model:    ${DRAFT_MODEL_PATH}"
echo "  Residual Model: ${RESIDUAL_MODEL_PATH}"
echo "  Benchmark:      ${BENCHMARK}"
echo "  Draft Lengths:  ${DRAFT_LENGTHS}"
echo "  Max Samples:    ${MAX_SAMPLES}"
echo "  Mode:           ${MODE}"
echo "  Output:         ${OUTPUT_FILE}"
echo "============================================================"

mkdir -p $(dirname ${OUTPUT_FILE})

python eval_speculative.py \
    --base_model_id ${MODEL_ID} \
    --draft_model_path ${DRAFT_MODEL_PATH} \
    --residual_model_path ${RESIDUAL_MODEL_PATH} \
    --benchmark ${BENCHMARK} \
    --max_samples ${MAX_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_lengths ${DRAFT_LENGTHS} \
    --mode ${MODE} \
    --output_file ${OUTPUT_FILE} \
    --device ${DEVICE}

echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "============================================================"
