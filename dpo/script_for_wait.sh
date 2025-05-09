#!/bin/bash

# 等待3小时
echo "脚本启动，将在3小时后开始实验: $(date)"
echo "预计开始时间: $(date -d '+3 hours')"
sleep 3h

# 记录开始时间
echo "开始执行实验: $(date)"

# 执行实验命令
PYTHONPATH=/cmlscratch/cheryunl/Avocado/dpo CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/examples/dpo/famo_dpo.py --sft_model_name "/cmlscratch/cheryunl/Avocado/sft/logs_trl/avocado/sft_famo_0.5" --dataset_name "Anthropic/hh-rlhf-mt"

# 记录结束时间和状态
if [ $? -eq 0 ]; then
  echo "实验成功完成: $(date)" | tee -a experiment_log.txt
else
  echo "实验运行出错，退出码: $?" | tee -a experiment_log.txt
fi