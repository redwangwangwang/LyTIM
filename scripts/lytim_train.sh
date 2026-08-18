#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-longitudinal-mimic}"
annotation="${ANNOTATION:-./dataset/mimic-cxr/annotation.json}"
base_dir="${BASE_DIR:-./dataset/mimic-cxr/}"
stage="lytim"
savepath="${SAVEPATH:-./save/${dataset}/${stage}}"
vision_model="${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}"
llama_model="${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}"
stage1_ckpt="${STAGE1_CKPT:-./save/${dataset}/stage1/checkpoints/best.pth}"
devices="${DEVICES:-2}"
strategy="${STRATEGY:-ddp}"

if [[ ! -f "${stage1_ckpt}" ]]; then
    echo "Stage-I checkpoint not found: ${stage1_ckpt}" >&2
    echo "Train Stage I first or set STAGE1_CKPT=/path/to/best.pth." >&2
    exit 1
fi
mkdir -p "${savepath}"

python -u train.py \
    --dataset "${dataset}" \
    --annotation "${annotation}" \
    --base_dir "${base_dir}" \
    --batch_size 2 \
    --val_batch_size 4 \
    --test_batch_size 4 \
    --savedmodel_path "${savepath}" \
    --delta_file "${stage1_ckpt}" \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 150 \
    --min_new_tokens 80 \
    --max_new_tokens 150 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices "${devices}" \
    --strategy "${strategy}" \
    --max_epochs 3 \
    --max_iteration 3 \
    --lytim_seed_source auto \
    --lytim_accept_epsilon 0.01 \
    --lytim_stop_energy 0.15 \
    --lytim_stop_threshold 0.5 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 0 \
    --vision_model "${vision_model}" \
    --llama_model "${llama_model}" \
    --accumulate_grad_batches 2 \
    --stage "${stage}" \
    2>&1 | tee -a "${savepath}/log.txt"
