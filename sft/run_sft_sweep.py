import wandb
import os
import subprocess

# 初始化sweep
sweep_id = wandb.sweep(sweep="sweep_config.yaml", project="Avocado")

# 定义agent运行函数
def agent_function():
    # 使用subprocess运行训练脚本
    cmd = f"python famo_sft.py --exp_type assistant_all"
    subprocess.run(cmd, shell=True)

# 启动agent
wandb.agent(sweep_id, function=agent_function, count=20)
