#!/usr/bin/env bash
# =====================================================================
# ControlNet-DPO 第二阶段训练: 在 SFT 权重基础上做偏好优化
# =====================================================================
# 前置条件:
#   1. 已完成 SFT 训练, controlnet 权重在
#        ./experiment/ControlNetSR/checkpoint-20000/controlnet
#   2. 已运行 scripts/build_preference.py 产出 candidates/ + score.json
# =====================================================================

set -euo pipefail

# GPU 配置 (按需调整)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# DPO 训练入口
accelerate launch --mixed_precision="fp16" train_controlnet.py \
    --config config/dpo.yaml \
    --train_method dpo \
    --beta_dpo 5000 \
    --sft_controlnet_ckpt ./experiment/ControlNetSR/checkpoint-20000/controlnet \
    --augment_geo \
    --null_text_ratio 0.0 \
    --enable_xformers_memory_efficient_attention true \
    --gradient_checkpointing \
    --mixed_precision "fp16" \
    --resolution 512 \
    --learning_rate 1e-6 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 2000 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 200 \
    --checkpointing_steps 500 \
    --output_dir "./experiment/ControlNetDPO" \
    --tracker_project_name "train_controlnet_DPO" \
    --dataloader_num_workers 0
