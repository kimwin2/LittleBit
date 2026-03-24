#!/bin/bash
# ==============================================================================
# CPU Speculative Decoding Benchmark
# ==============================================================================
#
# Measures:
#   1. FP16 target baseline speed via llama.cpp (autoregressive, CPU)
#   2. 0.1-bit draft speed via Python CPU kernel (autoregressive, CPU)
#   3. All-CPU speculative decoding (draft=CPU kernel, target=PyTorch CPU)
#
# The lb_kernels llama.cpp fork is NOT available, so we use:
#   - llama.cpp (upstream) for FP16 target speed measurement
#   - Python CPU kernel for draft speed + speculative decoding
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base FP16 model (HuggingFace format)
FP_MODEL_DIR="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# HF checkpoint (original trained output)
HF_CKPT_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit/2026_03_23_13_29"

# Runtime checkpoint (from convert_hf_to_runtime.py)
RUNTIME_CKPT_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.1bit/2026_03_23_13_29_runtime"

# GGUF for llama.cpp FP16 baseline
GGUF_DIR="/group-volume/ym1012.kim/homepc/LittleSpec/gguf_models"
FP_GGUF="${GGUF_DIR}/llama3.1-8b-instruct-f16.gguf"

# llama.cpp paths
LLAMA_CPP_DIR="lb_kernels/llama.cpp"
LLAMA_BUILD_DIR="${LLAMA_CPP_DIR}/build-cpu"

# Benchmark settings
THREADS=4
CTX=512
GEN_TOKENS=64
PROMPT="Write a Python function to compute fibonacci numbers efficiently."

# SD settings
DRAFT_LENGTHS="4"
MAX_SAMPLES=5
MAX_NEW_TOKENS=64

# ===========================
# STEP 1: Build llama.cpp for CPU (if not already built)
# ===========================

echo "============================================================"
echo "STEP 1: Building llama.cpp for CPU"
echo "============================================================"

LLAMA_CLI="${LLAMA_BUILD_DIR}/bin/llama-cli"
if [ -f "${LLAMA_CLI}" ]; then
    echo "llama.cpp already built. Skipping."
else
    mkdir -p "${LLAMA_BUILD_DIR}"
    cd "${LLAMA_BUILD_DIR}"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
    cmake --build . -j$(nproc) --target llama-cli
    cd ../../..
fi

# ===========================
# STEP 2: Convert FP16 to GGUF (if not already done)
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
fi

# ===========================
# STEP 3: Convert HF to runtime (if not already done)
# ===========================

echo ""
echo "============================================================"
echo "STEP 3: Converting draft to runtime format"
echo "============================================================"

if [ -d "${RUNTIME_CKPT_DIR}" ]; then
    echo "Runtime checkpoint already exists. Skipping."
else
    python convert_hf_to_runtime.py \
        --input_path ${HF_CKPT_DIR} \
        --output_path ${RUNTIME_CKPT_DIR} \
        --format runtime
fi

# Build CPU extension
echo "Building CPU extension..."
cd lb_kernels/littlebit_kernels_cpu
python setup.py build_ext --inplace 2>/dev/null || echo "Already built"
cd ../..

# ===========================
# STEP 4: FP16 Target Baseline (llama.cpp)
# ===========================

echo ""
echo "============================================================"
echo "STEP 4: FP16 Target Baseline (llama.cpp, CPU)"
echo "============================================================"

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

# ===========================
# STEP 5: Draft Speed Benchmark (Python CPU kernel)
# ===========================

echo ""
echo "============================================================"
echo "STEP 5: 0.1-bit Draft Speed Benchmark (CPU kernel)"
echo "============================================================"

python -c "
import time, sys, torch
sys.path.insert(0, 'lb_kernels')
from cpu_draft_model import CPUDraftModel

model = CPUDraftModel(
    runtime_path='${RUNTIME_CKPT_DIR}',
    base_model_id='${FP_MODEL_DIR}',
)

# Warmup
dummy = torch.tensor([[128000]], dtype=torch.long)
model.generate_draft_tokens(dummy, draft_length=1, greedy=True)

# Benchmark: generate ${GEN_TOKENS} tokens autoregressively
input_ids = torch.tensor([[128000]], dtype=torch.long)
start = time.time()
tokens = []
for i in range(${GEN_TOKENS}):
    draft_tokens, _ = model.generate_draft_tokens(
        input_ids, draft_length=1, greedy=True
    )
    tok = draft_tokens[0].reshape(-1)[0].item()
    tokens.append(tok)
    next_id = torch.tensor([[tok]], dtype=torch.long)
    input_ids = torch.cat([input_ids, next_id], dim=1)

elapsed = time.time() - start
tps = ${GEN_TOKENS} / elapsed
print(f'Draft CPU kernel: {tps:.2f} t/s ({elapsed:.2f}s for ${GEN_TOKENS} tokens)')
"

# ===========================
# STEP 6: All-CPU Speculative Decoding
# ===========================

echo ""
echo "============================================================"
echo "STEP 6: All-CPU Speculative Decoding (draft=CPU kernel, target=PyTorch FP)"
echo "============================================================"

python eval_speculative.py \
    --base_model_id ${FP_MODEL_DIR} \
    --draft_model_path ${RUNTIME_CKPT_DIR} \
    --target_mode fp \
    --draft_device cpu_kernel \
    --benchmark mt_bench \
    --max_samples ${MAX_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_lengths ${DRAFT_LENGTHS} \
    --mode greedy \
    --output_file eval_results/speculative_all_cpu_eval.json \
    --device cpu

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
