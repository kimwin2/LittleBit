#!/bin/bash
# ==============================================================================
# Full Pipeline: Build llama.cpp + Convert + Run CPU Speculative Decoding
# ==============================================================================
#
# This script:
#   1. Builds llama.cpp for CPU
#   2. Converts FP16 target model to GGUF (F16)
#   3. Converts 0.1-bit draft runtime checkpoint to LittleBit GGUF
#   4. Runs speculative decoding benchmark via llama-speculative-simple
#
# Prerequisites:
#   - cmake, ninja (or make), gcc/g++
#   - Python with safetensors, torch, transformers
#   - Converted runtime checkpoint (from convert_hf_to_runtime.py)
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base FP16 model (HuggingFace format)
FP_MODEL_DIR="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# Runtime checkpoint (from convert_hf_to_runtime.py)
RUNTIME_CKPT_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit/2026_03_23_13_29_runtime"

# Output GGUF files
GGUF_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/gguf_models"
FP_GGUF="${GGUF_DIR}/llama3.1-8b-instruct-f16.gguf"
DRAFT_GGUF="${GGUF_DIR}/llama3.1-8b-instruct-littlebit-0.1bit.gguf"

# llama.cpp paths
LLAMA_CPP_DIR="lb_kernels/llama.cpp"
LLAMA_BUILD_DIR="${LLAMA_CPP_DIR}/build-cpu"

# Benchmark settings
THREADS=4
CTX=512
GEN_TOKENS=64
DRAFT_MAX=8
PROMPT="Write a Python function to compute fibonacci numbers efficiently."

# ===========================
# STEP 1: Build llama.cpp for CPU
# ===========================

echo "============================================================"
echo "STEP 1: Building llama.cpp for CPU"
echo "============================================================"

if [ -f "${LLAMA_BUILD_DIR}/bin/llama-speculative-simple" ]; then
    echo "llama.cpp already built. Skipping."
else
    mkdir -p "${LLAMA_BUILD_DIR}"
    cd "${LLAMA_BUILD_DIR}"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build . -j$(nproc) --target llama-cli llama-speculative-simple
    cd ../../..
    echo "llama.cpp build complete!"
fi

# Verify binaries
LLAMA_CLI="${LLAMA_BUILD_DIR}/bin/llama-cli"
LLAMA_SPEC="${LLAMA_BUILD_DIR}/bin/llama-speculative-simple"

if [ ! -f "${LLAMA_CLI}" ]; then
    echo "ERROR: llama-cli not found at ${LLAMA_CLI}"
    exit 1
fi
if [ ! -f "${LLAMA_SPEC}" ]; then
    echo "ERROR: llama-speculative-simple not found at ${LLAMA_SPEC}"
    exit 1
fi
echo "Binaries OK: ${LLAMA_CLI}, ${LLAMA_SPEC}"

# ===========================
# STEP 2: Convert FP16 model to GGUF
# ===========================

echo ""
echo "============================================================"
echo "STEP 2: Converting FP16 model to GGUF"
echo "============================================================"

mkdir -p "${GGUF_DIR}"

if [ -f "${FP_GGUF}" ]; then
    echo "FP16 GGUF already exists. Skipping."
else
    python ${LLAMA_CPP_DIR}/convert_hf_to_gguf.py \
        ${FP_MODEL_DIR} \
        --outfile ${FP_GGUF} \
        --outtype f16
    echo "FP16 GGUF saved to ${FP_GGUF}"
fi

# ===========================
# STEP 3: Convert LittleBit draft to GGUF
# ===========================

echo ""
echo "============================================================"
echo "STEP 3: Converting LittleBit draft to GGUF"
echo "============================================================"

# HF checkpoint (original trained output)
HF_CKPT_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit/2026_03_23_13_29"
CANONICAL_DIR="${GGUF_DIR}/canonical_tmp"

if [ -f "${DRAFT_GGUF}" ]; then
    echo "Draft GGUF already exists. Skipping."
else
    # Step 3a: Convert HF → canonical format (u_sign_packed keys)
    echo "Converting HF checkpoint to canonical format..."
    python convert_hf_to_runtime.py \
        --input_path ${HF_CKPT_DIR} \
        --output_path ${CANONICAL_DIR} \
        --format canonical

    # Step 3b: Convert canonical → GGUF
    echo "Converting canonical to GGUF..."
    python lb_kernels/tools/convert_littlebit_hf_to_gguf.py \
        --input_dir ${CANONICAL_DIR} \
        --output_file ${DRAFT_GGUF} \
        --outtype f16
    echo "Draft GGUF saved to ${DRAFT_GGUF}"
fi

# ===========================
# STEP 4: Run baselines
# ===========================

echo ""
echo "============================================================"
echo "STEP 4: Running benchmarks"
echo "============================================================"

export LLAMA_LITTLEBIT_CPU_FUSED=1

echo ""
echo "--- FP16 Target baseline (autoregressive) ---"
${LLAMA_CLI} \
    -m ${FP_GGUF} \
    -ngl 0 \
    -t ${THREADS} \
    -c ${CTX} \
    -n ${GEN_TOKENS} \
    -p "${PROMPT}" \
    -s 0 \
    --temp 0 --top-k 1 \
    --ignore-eos \
    -st --simple-io

echo ""
echo "--- Draft 0.1-bit baseline (autoregressive) ---"
${LLAMA_CLI} \
    -m ${DRAFT_GGUF} \
    -ngl 0 \
    -t ${THREADS} \
    -c ${CTX} \
    -n ${GEN_TOKENS} \
    -p "${PROMPT}" \
    -s 0 \
    --temp 0 --top-k 1 \
    --ignore-eos \
    -st --simple-io

# ===========================
# STEP 5: Run speculative decoding
# ===========================

echo ""
echo "--- Speculative Decoding (draft=0.1bit, target=FP16) ---"
${LLAMA_SPEC} \
    -m ${FP_GGUF} \
    -md ${DRAFT_GGUF} \
    -ngl 0 \
    -t ${THREADS} \
    -c ${CTX} \
    -cd ${CTX} \
    -n ${GEN_TOKENS} \
    -p "${PROMPT}" \
    -s 0 \
    --temp 0 --top-k 1 \
    --ignore-eos \
    --draft-max ${DRAFT_MAX} \
    --draft-min 0 \
    --draft-p-min 0.0

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
