import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel, PeftConfig
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, check_lora_in_model_path, Instructions, Instructions_summary, \
                   build_dataset_eval, build_dataset_summary_eval, get_clean_data
tqdm.pandas()

def reward_guided_generate(
    model, 
    reward_models, 
    input_ids, 
    attention_mask, 
    instructions,
    tokenizer,
    preference_weights=None,
    beta=1.5,   # reward影响系数
    topk=10,    # 考虑的候选token数量
    **generation_kwargs
):
    """
    使用多个reward models实时引导生成的函数
    """
    print(f"DEBUG: reward_guided_generate called with beta={beta}, topk={topk}")
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 如果没有提供权重，默认平均分配
    if preference_weights is None:
        preference_weights = [1.0 / reward_models.num_rewards] * reward_models.num_rewards
    else:
        # 归一化权重
        total = sum(preference_weights)
        preference_weights = [w / total for w in preference_weights]
    
    print(f"DEBUG: Using preference weights: {preference_weights}")

    curr_input_ids = input_ids.clone()
    curr_attention_mask = attention_mask.clone()

    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    max_length = curr_input_ids.size(1) + generation_kwargs.get("max_new_tokens", 128)

    cached_output = None
    
    for step in range(max_length - curr_input_ids.size(1)):
        if not unfinished.any():
            break

        with torch.no_grad():
            # 获取模型输出
            if cached_output is None:
                model_outputs = model(
                    input_ids=curr_input_ids,
                    attention_mask=curr_attention_mask,
                    use_cache=True
                )
                cached_output = model_outputs.past_key_values
            else:
                model_outputs = model(
                    input_ids=curr_input_ids[:, -1].unsqueeze(-1),
                    attention_mask=curr_attention_mask,
                    past_key_values=cached_output,
                    use_cache=True
                )
                cached_output = model_outputs.past_key_values

            logits = model_outputs.logits[:, -1, :]

            # 应用温度参数
            if "temperature" in generation_kwargs and generation_kwargs["temperature"] > 0:
                logits = logits / generation_kwargs["temperature"]

            # 获取top-k candidates
            top_logits, top_indices = torch.topk(logits, topk, dim=-1)
            
            # 应用top-p过滤（正确实现）
            if "top_p" in generation_kwargs and generation_kwargs["top_p"] < 1.0:
                top_p = generation_kwargs["top_p"]
                
                # 分别处理每个batch，避免之前的bug
                filtered_indices = []
                filtered_logits = []
                
                for b in range(batch_size):
                    # 计算累积概率
                    batch_probs = torch.softmax(top_logits[b], dim=-1)
                    cumulative_probs = torch.cumsum(batch_probs, dim=-1)
                    
                    # 找到需要保留的token
                    mask = cumulative_probs < top_p
                    mask[0] = True  # 至少保留最高概率的token
                    
                    batch_top_indices = top_indices[b][mask]
                    batch_top_logits = top_logits[b][mask]
                    
                    filtered_indices.append(batch_top_indices)
                    filtered_logits.append(batch_top_logits)
            else:
                # 如果没有top-p，直接使用原始结果
                filtered_indices = [top_indices[b] for b in range(batch_size)]
                filtered_logits = [top_logits[b] for b in range(batch_size)]
            
            next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for b in range(batch_size):
                if not unfinished[b]:
                    # 如果已经finished，使用EOS token
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue

                current_candidates = filtered_indices[b]
                current_logits = filtered_logits[b]
                
                # 检查候选tokens是否为空
                if len(current_candidates) == 0:
                    # 如果没有候选tokens，使用EOS token结束生成
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue
                
                # 为每个候选token生成完整的候选序列
                candidate_input_ids = []
                for t in range(len(current_candidates)):
                    candidate = torch.cat([
                        curr_input_ids[b:b+1],
                        current_candidates[t:t+1].unsqueeze(0)
                    ], dim=1)
                    candidate_input_ids.append(candidate)
                
                candidate_batch = torch.cat(candidate_input_ids, dim=0)
                candidate_texts = tokenizer.batch_decode(candidate_batch, skip_special_tokens=True)

                # 提取query和response
                queries_responses = []
                for text in candidate_texts:
                    query = instructions.get_input(text)
                    response = instructions.get_response(text)
                    queries_responses.append((query, response))

                # 获取reward分数
                if hasattr(instructions, 'get_post'):
                    all_rewards = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
                else:
                    all_rewards = reward_models.get_reward_model_scores(queries_responses)

                # 计算加权rewards
                weighted_rewards = torch.zeros(len(candidate_input_ids), device=device)
                for i in range(reward_models.num_rewards):
                    if isinstance(all_rewards[i], list) or isinstance(all_rewards[i], tuple):
                        reward_tensor = torch.tensor(all_rewards[i], device=device)
                    else:
                        reward_tensor = all_rewards[i]
                    if reward_tensor.dim() > 1:
                        reward_tensor = reward_tensor.squeeze()
                    
                    # 确保reward tensor长度正确
                    if len(reward_tensor) != len(candidate_input_ids):
                        print(f"Warning: Reward tensor length {len(reward_tensor)} doesn't match candidate count {len(candidate_input_ids)}")
                        # 截断或填充到正确长度
                        if len(reward_tensor) > len(candidate_input_ids):
                            reward_tensor = reward_tensor[:len(candidate_input_ids)]
                        else:
                            # 如果不够，用最后一个值填充
                            padding = torch.full((len(candidate_input_ids) - len(reward_tensor),), 
                                                reward_tensor[-1] if len(reward_tensor) > 0 else 0.0, 
                                                device=device)
                            reward_tensor = torch.cat([reward_tensor, padding])
                    
                    weighted_rewards += preference_weights[i] * reward_tensor

                # 结合model logits和reward分数
                combined_scores = current_logits[:len(candidate_input_ids)] + beta * weighted_rewards

                # 根据是否采样来选择下一个token
                if generation_kwargs.get("do_sample", True):
                    sampling_probs = F.softmax(combined_scores / generation_kwargs.get("temperature", 1.0), dim=0)
                    next_token_idx = torch.multinomial(sampling_probs, num_samples=1)[0]
                else:
                    next_token_idx = torch.argmax(combined_scores)

                # 选择相应的token
                if next_token_idx < len(current_candidates):
                    next_tokens[b] = current_candidates[next_token_idx]
                else:
                    # fallback: 使用第一个候选token
                    next_tokens[b] = current_candidates[0] if len(current_candidates) > 0 else tokenizer.eos_token_id

            # 更新输入序列
            next_tokens = next_tokens.unsqueeze(1)
            curr_input_ids = torch.cat([curr_input_ids, next_tokens], dim=1)
            curr_attention_mask = torch.cat([
                curr_attention_mask, 
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            # 更新完成状态
            eos_mask = next_tokens.squeeze(1) == tokenizer.eos_token_id
            unfinished = unfinished & ~eos_mask
    
    return curr_input_ids


# 定义数据集路径
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/sft/logs_trl/avocado/sft_famo_0.5')
    dpo_model_path: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/dpo/output/dev/dpo/best_checkpoint')
    wandb_name: Optional[str] = field(default='eval_reward_guided', metadata={"help": "Name for this experiment"})
    reward_names: Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    
    # Reward guidance 相关参数
    beta: Optional[float] = field(default=1.5, metadata={"help": "beta parameter for reward influence"})
    topk: Optional[int] = field(default=10, metadata={"help": "topk parameter for candidate tokens"})
    preference_weights: Optional[str] = field(default="0.5,0.5", metadata={"help": "comma-separated weights for reward models"})
    use_reward_guidance: Optional[bool] = field(default=True, metadata={"help": "whether to use reward-guided generation"})
    
    # 生成参数
    num_samples: Optional[int] = field(default=0, metadata={"help": "Number of samples to evaluate (0 for all)"})
    generation_top_k: Optional[float] = field(default=0.0, metadata={"help": "top_k for generation"})
    generation_top_p: Optional[float] = field(default=0.9, metadata={"help": "top_p for generation"})
    generation_temperature: Optional[float] = field(default=1.0, metadata={"help": "temperature for generation"})
    generation_do_sample: Optional[bool] = field(default=True, metadata={"help": "whether to use sampling"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# 解析和设置参数
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
dpo_model_path = script_args.dpo_model_path
tokenizer_name = script_args.base_model_name

# 解析preference weights
preference_weights = [float(x.strip()) for x in script_args.preference_weights.split(',')]
print(f"Preference weights: {preference_weights}")

# 设置模型路径
model_path = dpo_model_path if (dpo_model_path is not None and os.path.exists(dpo_model_path)) else base_model_name
model_type = "DPO" if model_path == dpo_model_path else "SFT"
print(f"Using {model_type} model for evaluation: {model_path}")

# Accelerator设置
process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

# 设置reward models
reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(f"Reward models: {reward_names}")

reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError(f"Reward model {name} not found in dictionary")
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

# 设置随机种子
set_seed(8888)

# 加载tokenizer
tokenizer = load_main_tokenizer(tokenizer_name)

# 加载模型
if os.path.exists(os.path.join(model_path, "adapter_config.json")):
    print("Loading model as PEFT adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    peft_config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    model.resize_token_embeddings(len(tokenizer))

if hasattr(model, 'merge_and_unload'):
    print("Merging and unloading adapters...")
    model = model.merge_and_unload()

# 设置生成参数
generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": script_args.generation_top_k,
    "top_p": script_args.generation_top_p, 
    "do_sample": script_args.generation_do_sample,
    "temperature": script_args.generation_temperature,
}

print(f"Generation kwargs: {generation_kwargs}")

# 准备数据
print('Preparing evaluation dataset...')
tokenizer.padding_side = "left"

if exp_type == 'assistant':
    # 设置样本数量
    size = script_args.num_samples if script_args.num_samples > 0 else None
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, 
                                     reward_models.rm_tokenizers[0], 
                                     reward_models.rm_tokenizers[1], 
                                     split='test', size=size) 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, 
                                              reward_models.rm_tokenizers[0], 
                                              reward_models.rm_tokenizers[1], 
                                              split='test') 
    instructions = Instructions_summary()

print(f"Size of the validation set: {len(valid_dataset)}")

# 准备数据加载器
valid_batch_size = 1
remove_keys = []
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        remove_keys.append(key)
valid_dataset = valid_dataset.remove_columns(remove_keys)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, collate_fn=data_collator)
accelerator = Accelerator()
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

# 生成和评估
full_response_tensors = []
full_prompts = []

pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
print(f"\nStarting evaluation with reward guidance: {script_args.use_reward_guidance}")
print(f"Parameters: beta={script_args.beta}, topk={script_args.topk}")

with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        # 根据参数决定是否使用reward引导生成
        if script_args.use_reward_guidance:
            response_tensors = reward_guided_generate(
                accelerator.unwrap_model(model),
                reward_models,
                batch['input_ids'], 
                batch['attention_mask'],
                instructions,
                tokenizer,
                preference_weights=preference_weights,
                beta=script_args.beta,
                topk=script_args.topk,
                **generation_kwargs
            )
        else:
            # 使用原始的generate函数
            response_tensors = accelerator.unwrap_model(model).generate(
                batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                **generation_kwargs
            )
            
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

# 处理生成结果
full_prompts = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_responses = get_clean_data(full_responses, full_prompts)

# 计算reward分数
queries_responses = [
    (instructions.get_input(text), instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)

# 合并所有进程的结果
all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)

# 保存结果（只在主进程中）
if process_id == 0:
    evaluation_result = {
        'prompt': all_full_prompts,
        'response': all_full_responses,
    }
    
    # 添加每个reward model的得分
    for i, reward_name in enumerate(reward_names):
        evaluation_result[f'score_{reward_name}'] = all_rewards[i]
        print(f'Average {reward_name} score: {np.mean(all_rewards[i])}')
    
    # 设置保存文件名
    if script_args.use_reward_guidance:
        filename = f'reward_guided_eval_beta{script_args.beta}_topk{script_args.topk}_samples{script_args.num_samples}.csv'
    else:
        filename = f'baseline_eval_samples{script_args.num_samples}.csv'
    
    # 保存结果
    save_path = os.path.join(script_args.save_directory, script_args.wandb_name, filename)
    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    
    # 打印汇总统计
    print("\n=== Evaluation Summary ===")
    print(f"Model: {model_type}")
    print(f"Reward guidance: {script_args.use_reward_guidance}")
    if script_args.use_reward_guidance:
        print(f"Beta: {script_args.beta}, TopK: {script_args.topk}")
        print(f"Preference weights: {preference_weights}")
    print(f"Samples evaluated: {len(all_full_prompts)}")
    print("Average scores:")
    for i, reward_name in enumerate(reward_names):
        print(f"  {reward_name}: {np.mean(all_rewards[i]):.4f} (±{np.std(all_rewards[i]):.4f})")