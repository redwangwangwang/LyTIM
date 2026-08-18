#!/usr/bin/env bash
set -euo pipefail

dataset="${DATASET:-longitudinal-mimic}"
annotation="${ANNOTATION:-./dataset/mimic-cxr/annotation.json}"
base_dir="${BASE_DIR:-./dataset/mimic-cxr/}"
vision_model="${VISION_MODEL:-./pretrain_weights/swin-base-patch4-window7-224/}"
llama_model="${LLAMA_MODEL:-./pretrain_weights/Llama-2-7b-chat-hf/}"
lytim_ckpt="${LYTIM_CKPT:-./save/${dataset}/lytim/checkpoints/best.pth}"
stage1_ckpt="${STAGE1_CKPT:-./save/${dataset}/stage1/checkpoints/best.pth}"
savepath="${SAVEPATH:-./save/${dataset}/lytim}"

if [[ ! -f "${lytim_ckpt}" ]]; then
    echo "LyTIM checkpoint not found: ${lytim_ckpt}" >&2
    exit 1
fi

python -u train.py \
    --test \
    --dataset "${dataset}" \
    --annotation "${annotation}" \
    --base_dir "${base_dir}" \
    --savedmodel_path "${savepath}" \
    --ckpt_file "${lytim_ckpt}" \
    --delta_file "${stage1_ckpt}" \
    --stage lytim \
    --devices 1 \
    --strategy auto \
    --test_batch_size 4 \
    --max_iteration 3 \
    --vision_model "${vision_model}" \
    --llama_model "${llama_model}"
