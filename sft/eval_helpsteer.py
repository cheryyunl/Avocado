import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
import torch
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_main_tokenizer, check_lora_in_model_path

# 定义HelpSteer的属性名称
HELPSTEER_ATTRIBUTES = [
    'helpsteer-helpfulness', 
    'helpsteer-correctness', 
    'helpsteer-coherence',
    'helpsteer-complexity'
]

def build_helpsteer_eval_dataset(helpsteer_path, tokenizer, split='validation', seed=42, size=None):
    """HelpSteer评估数据集加载函数，与原始eval风格一致"""
    # 加载各子集
    ds_helpfulness = load_dataset(helpsteer_path, data_dir="helpfulness-positive", split=split)
    ds_correctness = load_dataset(helpsteer_path, data_dir="correctness-positive", split=split)
    ds_coherence = load_dataset(helpsteer_path, data_dir="coherence-positive", split=split)
    ds_complexity = load_dataset(helpsteer_path, data_dir="complexity-negative", split=split)
    
    # 合并所有子集
    ds = concatenate_datasets([ds_helpfulness, ds_correctness, ds_coherence, ds_complexity])
    ds = ds.shuffle(seed=seed)
    
    # 限制数据集大小（如果指定）
    if size is not None:
        ds = ds.select(range(size))
    
    # 样本处理函数 - 直接参考原始build_dataset_eval的风格
    def tokenize(sample):
        # 创建prompt格式
        sample['prompt'] = f"Human: {sample['prompt']}\nAssistant: "
        # 保存原始prompt用于后续评估
        sample['original_prompt'] = sample['prompt'].replace("Human: ", "").replace("\nAssistant: ", "")
        # 编码为input_ids
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["attention_mask"] = [1] * len(sample["input_ids"])
        return sample
    
    # 应用处理函数 - 与原始代码保持一致的参数
    ds_processed = ds.map(tokenize, batched=False, num_proc=20)
    
    # 过滤过长或过短的样本
    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    
    # 移除不需要的列
    columns_to_remove = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity', 'response']
    columns_to_remove = [col for col in columns_to_remove if col in ds_processed.column_names]
    if columns_to_remove:
        ds_processed = ds_processed.remove_columns(columns_to_remove)
    
    # 设置为torch格式 - 关键步骤
    ds_processed.set_format(type="torch")
    
    return ds_processed

class HelpSteerRewardModel:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        print(f"Loading HelpSteer reward model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=True
        )
        self.num_rewards = len(HELPSTEER_ATTRIBUTES)
        print(f"Loaded HelpSteer model with {self.num_rewards} attributes")
        
    def get_reward_scores(self, query_response_pairs):
        """评估一批query-response对，返回每个属性的评分"""
        all_scores = [[] for _ in range(self.num_rewards)]
        
        for prompt, response in tqdm(query_response_pairs, desc="Evaluating responses"):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids)
                # 获取四个属性评分
                for i in range(self.num_rewards):
                    all_scores[i].append(output.score[i].cpu().float().item())
        
        return all_scores

def get_clean_response(full_text, prompt):
    """从生成的文本中提取助手的回复"""
    if full_text.startswith(prompt):
        response = full_text[len(prompt):]
    else:
        if "Assistant:" in full_text:
            response = full_text.split("Assistant:", 1)[1]
        else:
            response = full_text
    
    return response.strip()

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='./huggingface_models/Llama-2-7b-hf')
    wandb_name: Optional[str] = field(default='eval_helpsteer', metadata={"help": "Name for this experiment"})
    dataset_path: Optional[str] = field(default='cheryyunl/helpsteer', metadata={"help": "Dataset to evaluate on"})
    reward_model_path: Optional[str] = field(default='nicolinho/QRM-Llama3.1-8B-v2')
    num_samples: Optional[int] = field(default=400, metadata={"help": "Total number of samples to evaluate (0 for all)"})
    split: Optional[str] = field(default='validation', metadata={"help": "Dataset split to use"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    base_model_name = script_args.base_model_name
    tokenizer_name = script_args.base_model_name
    print('Base model: ', base_model_name)
    
    # 获取进程ID和GPU配置
    accelerator = Accelerator()
    process_id = accelerator.local_process_index 
    num_processes = accelerator.num_processes
    gpu_id = process_id
    print(f'Process: {process_id}/{num_processes}, model GPU ID: {gpu_id}')
    
    # 创建输出目录
    os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)
    
    # 设置随机种子
    set_seed(8888)
    
    # 加载tokenizer和模型
    tokenizer = load_main_tokenizer(tokenizer_name)
    tokenizer.padding_side = "left"  # 重要：设置padding方向
    
    # 使用accelerate来加载模型到适当的GPU
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map={"": gpu_id}
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # 检查是否使用了LoRA
    if check_lora_in_model_path(model, base_model_name):
        model = PeftModel.from_pretrained(model, base_model_name)
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    
    # 加载HelpSteer reward模型
    reward_model = HelpSteerRewardModel(script_args.reward_model_path, device=f"cuda:{gpu_id}")
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": 128,
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }
    
    # 加载评估数据集
    print('Loading evaluation dataset...')
    # 如果设置了num_samples，直接在加载时限制大小
    size = script_args.num_samples if script_args.num_samples > 0 else None
    valid_dataset = build_helpsteer_eval_dataset(
        script_args.dataset_path, 
        tokenizer, 
        split=script_args.split,
        size=size
    )
    
    print(f"Size of the validation set: {len(valid_dataset)}")
    
    # 批处理大小设为1以避免问题
    valid_batch_size = 1
    
    # 使用与原始评估相同的数据加载方式
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=valid_batch_size, 
        drop_last=False
    )
    
    # 使用accelerator准备模型和数据加载器
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)
    
    # 评估循环
    full_response_tensors = []
    full_prompts = []
    original_prompts = []
    
    total_steps = len(valid_dataset) // valid_batch_size // num_processes + 1
    pbar = tqdm(total=total_steps, desc=f"GPU {gpu_id} generating")
    
    # 生成回复
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            # 保存原始prompt
            if 'original_prompt' in batch:
                original_prompts.extend(batch['original_prompt'])
            
            # 生成回复
            response_tensors = accelerator.unwrap_model(model).generate(
                batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                **generation_kwargs
            )
            
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    # 解码生成的文本
    input_texts = tokenizer.batch_decode(full_prompts, skip_special_tokens=True)
    full_responses = tokenizer.batch_decode(full_response_tensors, skip_special_tokens=True)
    
    # 提取助手回复部分
    clean_responses = []
    for full_text, prompt in zip(full_responses, input_texts):
        response = get_clean_response(full_text, prompt)
        clean_responses.append(response)
    
    # 准备评估数据
    query_response_pairs = list(zip(original_prompts, clean_responses))
    
    # 获取每个属性的评分
    print(f"GPU {gpu_id} evaluating responses...")
    rewards_list = reward_model.get_reward_scores(query_response_pairs)
    
    # 使用accelerator收集所有进程的结果
    all_rewards = []
    for i in range(reward_model.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    
    all_full_prompts = accelerator.gather_for_metrics(original_prompts)
    all_full_responses = accelerator.gather_for_metrics(clean_responses)
    
    # 只在主进程保存结果
    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        
        # 添加每个属性的评分
        for i, attr_name in enumerate(HELPSTEER_ATTRIBUTES):
            evaluation_result[attr_name] = all_rewards[i]
            print(f'Average {attr_name} score: {np.mean(all_rewards[i])}')
        
        # 计算整体平均分
        all_scores_array = np.array([all_rewards[i] for i in range(len(HELPSTEER_ATTRIBUTES))])
        overall_scores = np.mean(all_scores_array, axis=0)
        evaluation_result['overall_score'] = overall_scores
        print(f'Overall average score: {np.mean(overall_scores)}')
        
        # 保存结果到CSV
        dataframe = pd.DataFrame(evaluation_result)
        filename = os.path.join(
            script_args.save_directory, 
            script_args.wandb_name,
            'helpsteer_eval_results.csv'
        )
        dataframe.to_csv(filename)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()