dataset="longitudinal-mimic"
annotation="./dataset/mimic-cxr/annotation.json"
base_dir="./dataset/mimic-cxr/"

stage="stage2"
savepath="./save/$dataset/$stage"

vision_model="./pretrain_weights/swin-base-patch4-window7-224/"
llama_model="./pretrain_weights/Llama-2-7b-chat-hf/"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 2 \
    --val_batch_size 4 \
    --freeze_vm False \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 150 \
    --min_new_tokens 80 \
    --max_new_tokens 150 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 2 \
    --max_epochs 3 \
    --limit_val_batches 1.0 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --vision_model ${vision_model} \
    --llama_model ${llama_model} \
    --longitudinal True \
    --accumulate_grad_batches 2 \
    --stage stage1 \
    2>&1 |tee -a ${savepath}/log.txt
