#!/bin/bash
# ==============================================================================
# Step 2 Only: Train 1.7-bit Residual Model (skip Step 1)
# ==============================================================================
#
# Use this when Step 1 (0.3-bit draft model) already completed successfully
# but Step 2 failed or needs to be re-run.
#
# Step 1 output: /group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit_200k/2026_03_31_18_15
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model (same as the crashed pipeline)
MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Dataset
DATASET="openhermes"
DATA_ROOT="./"
NUM_SAMPLES=200000

# Draft model from completed Step 1
DRAFT_MODEL_PATH="/group-volume/ym1012.kim/homepc/LittleSpec/outputs/step1_draft_0.3bit_200k/2026_03_31_18_15"

# Output directories (must match the original pipeline structure)
STEP1_SAVE_DIR="outputs/step1_draft_0.3bit_200k"
STEP2_SAVE_DIR="outputs/step2_residual_1.7bit_200k"

# Quantization
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Step-specific bit widths (must match the original pipeline)
STEP1_EFF_BIT=0.3
STEP2_EFF_BIT=1.7

# Training (same as crashed pipeline)
NUM_EPOCHS=5
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=4e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=1.0

# Training-time test
TRAIN_TIME_TEST="false"
TTT_STEPS=7
TTT_DECAY=0.8

# DeepSpeed (original used 8 GPUs)
NUM_GPUS=8
DS_CONFIG="configs/zero3.json"

# Logging
STEP1_RUN_NAME="step1_draft_0.3bit_200k"
STEP2_RUN_NAME="step2_residual_1.7bit_200k"
REPORT="tensorboard"

# ===========================
# VALIDATE
# ===========================

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
    echo "ERROR: Draft model directory not found: ${DRAFT_MODEL_PATH}"
    echo "  Step 1 must be completed first."
    exit 1
fi

if [ ! -f "${DRAFT_MODEL_PATH}/littlebit_config.json" ]; then
    echo "ERROR: No littlebit_config.json found in ${DRAFT_MODEL_PATH}"
    echo "  This doesn't look like a valid Step 1 output."
    exit 1
fi

echo "Draft model config:"
cat "${DRAFT_MODEL_PATH}/littlebit_config.json"
echo ""

# ===========================
# LAUNCH (Step 2 only)
# ===========================

echo "============================================================"
echo "Step 2 Only: Train ${STEP2_EFF_BIT}-bit Residual Model"
echo "  Model:       ${MODEL_ID}"
echo "  Dataset:     ${DATASET} (${NUM_SAMPLES} samples)"
echo "  Draft model: ${DRAFT_MODEL_PATH}"
echo "  Step1 bit:   ${STEP1_EFF_BIT}"
echo "  Step2 bit:   ${STEP2_EFF_BIT}"
echo "  Output:      ${STEP2_SAVE_DIR}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Epochs:      ${NUM_EPOCHS}"
echo "  LR:          ${LR}"
echo "  L2L scale:   ${L2L_LOSS_SCALE}"
echo "============================================================"

deepspeed --num_gpus=${NUM_GPUS} train_full_pipeline.py \
    --model_id ${MODEL_ID} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --num_samples ${NUM_SAMPLES} \
    --step1_save_dir ${STEP1_SAVE_DIR} \
    --step2_save_dir ${STEP2_SAVE_DIR} \
    --step1_eff_bit ${STEP1_EFF_BIT} \
    --step2_eff_bit ${STEP2_EFF_BIT} \
    --quant_func ${QUANT_FUNC} \
    --quant_mod ${QUANT_MOD} \
    --residual ${RESIDUAL} \
    --kv_factor ${KV_FACTOR} \
    --min_split_dim ${MIN_SPLIT_DIM} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACC_STEPS} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --l2l_loss_scale ${L2L_LOSS_SCALE} \
    --ds_config_path ${DS_CONFIG} \
    --step1_run_name ${STEP1_RUN_NAME} \
    --step2_run_name ${STEP2_RUN_NAME} \
    --report ${REPORT} \
    --train_time_test ${TRAIN_TIME_TEST} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_decay ${TTT_DECAY} \
    --skip_step1 \
    --draft_model_path ${DRAFT_MODEL_PATH}

echo "============================================================"
echo "Step 2 Complete!"
echo "  Residual model: ${STEP2_SAVE_DIR}/"
echo "  Draft model:    ${DRAFT_MODEL_PATH}"
echo "============================================================"
