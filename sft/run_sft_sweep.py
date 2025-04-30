import wandb
import subprocess
import os

# 定义sweep配置
sweep_config = {
    'method': 'bayes',  
    'metric': {
        'name': 'multi_objective_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'w_lr': {'values': [0.001, 0.005, 0.01, 0.05]},
        'gamma': {'values': [0.001, 0.01, 0.1]},
        'famo_update_frequency': {'values': [5, 10, 20]},
        'ema_alpha': {'values': [0.85, 0.9, 0.95]},
        'init_steps': {'values': [100, 200, 500]},
        'loss_scale': {'values': ["{1: 1, 3: 0.4}", "{1: 1, 3: 0.5}", "{1: 1, 3: 0.6}"]},
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.00005
        }
    }
}

# 初始化sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="Avocado")

# 定义agent运行函数
def agent_function():
    # 使用torchrun运行分布式训练
    cmd = "torchrun --nproc_per_node=4 famo_sft.py --exp_type assistant_all"
    subprocess.run(cmd, shell=True)

# 启动agent
wandb.agent(sweep_id, function=agent_function, count=20)  # 运行10次试验