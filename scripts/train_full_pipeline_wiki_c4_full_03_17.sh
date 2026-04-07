#!/bin/bash
# ==============================================================================
# Full Pipeline: Train 0.3-bit Draft + 1.7-bit Residual
#   Dataset: WikiText2 (full) + C4 shard 0 (full, 100%)
# ==============================================================================
#
# Unlike mixed_regen which uses OpenHermes + WikiText2 + C4 (50%),
# this script uses only raw text data:
#   - WikiText2 train: full split
#   - C4 shard 0: full (100%, not 50%)
#
# Pipeline:
#   Step 1: Train 0.3-bit draft model (QAT with KD)
#   Step 2: Train 1.7-bit residual model (Matryoshka)
#
# ==============================================================================

set -e

# ===========================
# USER CONFIGURATION
# ===========================

# Model
MODEL_ID="/group-volume/ym1012.kim/homepc/LittleSpec/Llama-3.1-8B-Instruct"

# Dataset: wiki_c4_full = WikiText2 full + C4 shard 0 full (100%)
DATASET="wiki_c4_full"
DATA_ROOT="./"

# Output directories
STEP1_SAVE_DIR="outputs/step1_draft_0.3bit_wiki_c4_full"
STEP2_SAVE_DIR="outputs/step2_residual_1.7bit_wiki_c4_full"

# Quantization (shared)
QUANT_FUNC="STEBinary"
QUANT_MOD="LittleBitLinear"
RESIDUAL="false"
KV_FACTOR=1.0
MIN_SPLIT_DIM=8

# Step-specific bit widths
STEP1_EFF_BIT=0.3
STEP2_EFF_BIT=1.7

# Training
NUM_EPOCHS=5
BATCH_SIZE=4
GRAD_ACC_STEPS=1
LR=4e-5
WARMUP_RATIO=0.03
L2L_LOSS_SCALE=1.0

# Training-time test (Step 1 only)
TRAIN_TIME_TEST="false"
TTT_STEPS=7
TTT_DECAY=0.8

# DeepSpeed
NUM_GPUS=8
DS_CONFIG="configs/zero3.json"

# Logging
STEP1_RUN_NAME="step1_draft_0.3bit_wiki_c4_full"
STEP2_RUN_NAME="step2_residual_1.7bit_wiki_c4_full"
REPORT="tensorboard"

# Pipeline control
SKIP_STEP1="false"
SKIP_STEP2="false"
DRAFT_MODEL_PATH=""

# ===========================
# LAUNCH TRAINING PIPELINE
# ===========================

echo ""
echo "============================================================"
echo "Full Pipeline: Matryoshka Training (WikiText2 + C4 Full)"
echo "  Model:    ${MODEL_ID}"
echo "  Dataset:  ${DATASET}"
echo "    - WikiText2:  full train split"
echo "    - C4:         first shard, 100% (full)"
echo "  Step 1:   ${STEP1_EFF_BIT}-bit draft  -> ${STEP1_SAVE_DIR}"
echo "  Step 2:   ${STEP2_EFF_BIT}-bit residual -> ${STEP2_SAVE_DIR}"
echo "  GPUs:     ${NUM_GPUS}"
echo "============================================================"

SKIP_STEP1_ARG=""
if [ "${SKIP_STEP1}" = "true" ]; then
    SKIP_STEP1_ARG="--skip_step1"
fi

SKIP_STEP2_ARG=""
if [ "${SKIP_STEP2}" = "true" ]; then
    SKIP_STEP2_ARG="--skip_step2"
fi

DRAFT_MODEL_ARG=""
if [ -n "${DRAFT_MODEL_PATH}" ]; then
    DRAFT_MODEL_ARG="--draft_model_path ${DRAFT_MODEL_PATH}"
fi

deepspeed --num_gpus=${NUM_GPUS} train_full_pipeline.py \
    --model_id ${MODEL_ID} \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
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
    --train_time_test ${TRAIN_TIME_TEST} \
    --ttt_steps ${TTT_STEPS} \
    --ttt_decay ${TTT_DECAY} \
    --ds_config_path ${DS_CONFIG} \
    --report ${REPORT} \
    --step1_run_name ${STEP1_RUN_NAME} \
    --step2_run_name ${STEP2_RUN_NAME} \
    ${SKIP_STEP1_ARG} \
    ${SKIP_STEP2_ARG} \
    ${DRAFT_MODEL_ARG}
