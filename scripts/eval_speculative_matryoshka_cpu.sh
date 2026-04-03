#!/bin/bash
# ==============================================================================
# All-CPU Speculative Decoding: 0.3-bit Draft + (0.3+1.7)-bit Target
# ==============================================================================
#
# Both draft and target models run on CPU using the LittleBit C++ kernel.
#
#   Draft:  0.3-bit quantized model  (CPUDraftModel, single LB model)
#   Target: 0.3-bit + 1.7-bit combined (CPUTargetModel, sums hidden states
#           from draft + residual LB models, then applies lm_head once)
#
# Measures: TPS, acceptance rate, mean acceptance length, speedup vs baseline.
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Base model (tokenizer + embedding/lm_head source)
BASE_MODEL_ID="/group-volume/ym1012.kim/homepc/EAGLE/Llama-3.1-8B-Instruct"

# --- Draft model (0.3-bit) ---
# HF checkpoint (for any HF-based needs, not directly used by CPU kernel)
DRAFT_HF_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01"
# Runtime checkpoint (converted for C++ kernel)
DRAFT_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit/2026_03_31_19_01_runtime"

# --- Target model components ---
# Draft component runtime (0.3-bit, same architecture, used by CPUTargetModel)
TARGET_DRAFT_RUNTIME_PATH="${DRAFT_RUNTIME_PATH}"
# Residual component runtime (1.7-bit)
RESIDUAL_RUNTIME_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step2_residual_1.7bit/2026_03_31_19_01_runtime"

# --- Runtime checkpoint conversion (if not done yet) ---
# The runtime checkpoints must exist. If not, convert:
#   python convert_hf_to_runtime.py \
#     --input_path <HF_CHECKPOINT_PATH> \
#     --output_path <RUNTIME_OUTPUT_PATH> \
#     --format runtime

# Benchmark settings
THREADS=4
DRAFT_LENGTHS="4"
MAX_SAMPLES=5
MAX_NEW_TOKENS=64
MODE="greedy"

# Output
OUTPUT_DIR="eval_results/speculative_matryoshka_cpu"
OUTPUT_FILE="${OUTPUT_DIR}/speculative_matryoshka_cpu_eval.json"

# Thread configuration
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}

# ===========================
# VALIDATE PATHS
# ===========================

echo "============================================================"
echo "Validating model paths..."
echo "============================================================"

for dir_path in "${DRAFT_RUNTIME_PATH}" "${RESIDUAL_RUNTIME_PATH}"; do
    if [ ! -d "${dir_path}" ]; then
        echo "ERROR: Directory not found: ${dir_path}"
        echo "  Please convert HF checkpoint to runtime format first."
        exit 1
    fi
done

echo "  Draft runtime:    ${DRAFT_RUNTIME_PATH} ✓"
echo "  Residual runtime: ${RESIDUAL_RUNTIME_PATH} ✓"

# ===========================
# APPLY OPTIMIZED KERNEL & REBUILD
# ===========================

echo ""
echo "============================================================"
echo "Applying optimized C++ kernel and cleaning build cache..."
echo "============================================================"

if [ -f "patches/littlebit_cpu_optimized.cpp" ]; then
    echo "  Copying patches/littlebit_cpu_optimized.cpp → lb_kernels/littlebit_kernels_cpu/littlebit_cpu.cpp"
    cp patches/littlebit_cpu_optimized.cpp lb_kernels/littlebit_kernels_cpu/littlebit_cpu.cpp
else
    echo "  No optimized patch found, using existing kernel."
fi

# Clean build cache to force recompilation
rm -rf lb_kernels/littlebit_kernels_cpu/build/
echo "  Build cache cleared."

# ===========================
# STEP 1: Draft-only Speed Benchmark
# ===========================

echo ""
echo "============================================================"
echo "STEP 1: 0.3-bit Draft-only Speed Benchmark (CPU kernel)"
echo "============================================================"

python -c "
import time, sys, torch
torch.set_num_threads(${THREADS})
print(f'PyTorch threads: {torch.get_num_threads()}', flush=True)
sys.path.insert(0, 'lb_kernels')
from cpu_draft_model import CPUDraftModel

model = CPUDraftModel(
    runtime_path='${DRAFT_RUNTIME_PATH}',
    base_model_id='${BASE_MODEL_ID}',
)

# Warmup
model.reset()
input_ids = torch.tensor([[128000]], dtype=torch.long)
model.prefill(input_ids)
for i in range(3):
    draft_tokens, _ = model.generate_draft_tokens(input_ids, draft_length=1, greedy=True)
    tok = draft_tokens[0].reshape(-1)[0].item()
    input_ids = torch.cat([input_ids, torch.tensor([[tok]], dtype=torch.long)], dim=1)

# Benchmark
model.reset()
input_ids = torch.tensor([[128000]], dtype=torch.long)
model.prefill(input_ids)

GEN_TOKENS = ${MAX_NEW_TOKENS}
start = time.time()
for i in range(GEN_TOKENS):
    draft_tokens, _ = model.generate_draft_tokens(input_ids, draft_length=1, greedy=True)
    tok = draft_tokens[0].reshape(-1)[0].item()
    input_ids = torch.cat([input_ids, torch.tensor([[tok]], dtype=torch.long)], dim=1)
    if (i+1) % 16 == 0:
        elapsed_so_far = time.time() - start
        print(f'  Token {i+1}/{GEN_TOKENS}: avg {(i+1)/elapsed_so_far:.2f} t/s', flush=True)

elapsed = time.time() - start
tps = GEN_TOKENS / elapsed
print(f'')
print(f'=== Draft CPU kernel result ===')
print(f'  Tokens: {GEN_TOKENS}')
print(f'  Time:   {elapsed:.2f}s')
print(f'  Speed:  {tps:.2f} t/s')
"

# ===========================
# STEP 2: Target-only Speed Benchmark
# ===========================

echo ""
echo "============================================================"
echo "STEP 2: (0.3+1.7)-bit Target Speed Benchmark (CPU kernel)"
echo "============================================================"

python -c "
import time, sys, torch
torch.set_num_threads(${THREADS})
print(f'PyTorch threads: {torch.get_num_threads()}', flush=True)
sys.path.insert(0, 'lb_kernels')
from cpu_target_model import CPUTargetModel

model = CPUTargetModel(
    draft_runtime_path='${TARGET_DRAFT_RUNTIME_PATH}',
    residual_runtime_path='${RESIDUAL_RUNTIME_PATH}',
    base_model_id='${BASE_MODEL_ID}',
)

# Benchmark: autoregressive with target model
GEN_TOKENS = ${MAX_NEW_TOKENS}
input_ids = torch.tensor([[128000]], dtype=torch.long)

start = time.time()
for i in range(GEN_TOKENS):
    logits = model.forward(input_ids)
    next_token = torch.argmax(logits[0, -1, :]).item()
    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
    if (i+1) % 8 == 0:
        elapsed_so_far = time.time() - start
        print(f'  Token {i+1}/{GEN_TOKENS}: avg {(i+1)/elapsed_so_far:.2f} t/s', flush=True)

elapsed = time.time() - start
tps = GEN_TOKENS / elapsed
print(f'')
print(f'=== Target CPU kernel result ===')
print(f'  Tokens: {GEN_TOKENS}')
print(f'  Time:   {elapsed:.2f}s')
print(f'  Speed:  {tps:.2f} t/s')
"

# ===========================
# STEP 3: All-CPU Speculative Decoding
# ===========================

echo ""
echo "============================================================"
echo "STEP 3: All-CPU Speculative Decoding"
echo "  Draft:  0.3-bit LittleBit CPU kernel"
echo "  Target: (0.3+1.7)-bit LittleBit CPU kernel"
echo "  K:      ${DRAFT_LENGTHS}"
echo "  Tokens: ${MAX_NEW_TOKENS}"
echo "  Samples: ${MAX_SAMPLES}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

python eval_speculative.py \
    --base_model_id "${BASE_MODEL_ID}" \
    --draft_model_path "${DRAFT_RUNTIME_PATH}" \
    --target_mode littlebit_cpu \
    --draft_runtime_path "${TARGET_DRAFT_RUNTIME_PATH}" \
    --residual_runtime_path "${RESIDUAL_RUNTIME_PATH}" \
    --draft_device cpu_kernel \
    --benchmark mt_bench \
    --max_samples ${MAX_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --draft_lengths ${DRAFT_LENGTHS} \
    --mode ${MODE} \
    --output_file "${OUTPUT_FILE}" \
    --device cpu

echo ""
echo "============================================================"
echo "All-CPU Speculative Decoding Complete!"
echo "  Results: ${OUTPUT_FILE}"
echo "============================================================"
