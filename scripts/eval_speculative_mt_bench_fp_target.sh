#!/bin/bash
# ==============================================================================
# MT-Bench Speculative Decoding Evaluation (FP Target Mode)
# ==============================================================================
#
# Step2(residual)를 하지 않은 상태에서 speculative decoding 평가:
#   - Draft model: 0.3-bit quantized (CPU kernel or GPU)
#   - Target model: Original FP16 model (base model 그대로)
#
# Measures TPS, acceptance rate, mean acceptance length.
# Uses MT-Bench 80 questions (Turn 1 only).
# No LLM judge.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base model (tokenizer + FP target model로 사용)
BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# Draft model: runtime format (converted for CPU kernel)
# >>> 0.3-bit draft 모델 경로를 설정하세요 <<<
# CPU kernel 사용 시:  convert_hf_to_runtime.py로 변환된 _runtime 경로
# GPU 사용 시:        HF format 체크포인트 경로 그대로
DRAFT_MODEL_PATH="outputs/step1_draft_0.3bit/REPLACE_WITH_TIMESTAMP_runtime"

# Target mode: "fp" = base model을 그대로 target으로 사용 (Step2 불필요)
TARGET_MODE="fp"

# Draft device: "cpu_kernel" for C++ LittleBit kernel, "cuda" for GPU
DRAFT_DEVICE="cpu_kernel"

# Speculative decoding params
DRAFT_LENGTH=5          # K: number of draft tokens per step
MAX_NEW_TOKENS=512
MAX_QUESTIONS=3         # Set to 80 for full eval, 3 for quick test

# Also run autoregressive baseline for speedup comparison?
RUN_BASELINE="true"

# Mode
MODE="greedy"

# Output
OUTPUT_DIR="eval_results/speculative_mt_bench_fp_target"

# ===========================
# VALIDATE
# ===========================

if [[ "${DRAFT_MODEL_PATH}" == *"REPLACE_WITH_TIMESTAMP"* ]]; then
    echo "ERROR: Please set DRAFT_MODEL_PATH to actual trained model directory."
    echo "  Edit this script and replace REPLACE_WITH_TIMESTAMP with actual path."
    exit 1
fi

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    exit 1
fi

# ===========================
# BUILD CPU EXTENSION (if needed)
# ===========================

if [ "${DRAFT_DEVICE}" = "cpu_kernel" ]; then
    echo "Building CPU extension..."
    cd lb_kernels/littlebit_kernels_cpu
    python setup.py build_ext --inplace 2>/dev/null || echo "CPU extension build skipped (may already be built)"
    cd ../..
fi

# ===========================
# RUN
# ===========================

echo "============================================================"
echo "MT-Bench Speculative Decoding Evaluation (FP Target)"
echo "  Base Model (= FP Target): ${BASE_MODEL_ID}"
echo "  Draft:       ${DRAFT_MODEL_PATH} (${DRAFT_DEVICE})"
echo "  Target Mode: ${TARGET_MODE} (original FP16, no Step2 needed)"
echo "  K:           ${DRAFT_LENGTH}"
echo "  Max Tokens:  ${MAX_NEW_TOKENS}"
echo "  Questions:   ${MAX_QUESTIONS}"
echo "  Baseline:    ${RUN_BASELINE}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

python eval_speculative_mt_bench.py \
    --base_model_id "${BASE_MODEL_ID}" \
    --draft_model_path "${DRAFT_MODEL_PATH}" \
    --target_mode "${TARGET_MODE}" \
    --draft_device "${DRAFT_DEVICE}" \
    --draft_length ${DRAFT_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_questions ${MAX_QUESTIONS} \
    --mode "${MODE}" \
    --run_baseline "${RUN_BASELINE}" \
    --output_dir "${OUTPUT_DIR}"

echo "============================================================"
echo "Evaluation Complete!"
echo "  Results: ${OUTPUT_DIR}/speculative_mt_bench_results.json"
echo "============================================================"
