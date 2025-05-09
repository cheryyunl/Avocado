#!/bin/bash

# 记录开始时间
echo "开始执行实验: $(date)"

# 执行实验命令
PYTHONPATH=/cmlscratch/cheryunl/Avocado/dpo CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/examples/dpo/famo_dpo.py --sft_model_name "/cmlscratch/cheryunl/Avocado/sft/logs_trl/avocado/sft_famo_0.5" --dataset_name "Anthropic/hh-rlhf-mt"

# 记录结束时间
echo "实验完成: $(date)"