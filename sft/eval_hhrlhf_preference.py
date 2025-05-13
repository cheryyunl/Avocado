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
    preference_weights=None,
    beta=2,   # reward影响系数
    topk=10,    # 考虑的候选token数量，基于论文使用k=10
    **generation_kwargs
):
    """
    使用多个reward models实时引导生成的函数
    """
    print(f"Reward_guided_generate called with beta={beta}, topk={topk}")
    print(f"Preference_weights={preference_weights}")
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 如果没有提供权重，默认平均分配
    if preference_weights is None:
        preference_weights = [1.0 / reward_models.num_rewards] * reward_models.num_rewards
    else:
        # 归一化权重
        total = sum(preference_weights)
        preference_weights = [w / total for w in preference_weights]

    curr_input_ids = input_ids.clone()
    curr_attention_mask = attention_mask.clone()

    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    max_length = curr_input_ids.size(1) + generation_kwargs.get("max_new_tokens", 128)

    cached_output = None
    step = 0
    
    for step in range(max_length - curr_input_ids.size(1)):
        if not unfinished.any():
            break

        with torch.no_grad():
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

            if "temperature" in generation_kwargs and generation_kwargs["temperature"] > 0:
                logits = logits / generation_kwargs["temperature"]
            
            # 获取 top-k 候选
            top_logits, top_indices = torch.topk(logits, topk, dim=-1)
            
            # 应用 top-p 过滤 - 修复版本！
            if "top_p" in generation_kwargs and generation_kwargs["top_p"] < 1.0:
                top_p = generation_kwargs["top_p"]
                
                # 为每个batch分别处理（避免之前的错误）
                filtered_indices = []
                filtered_logits = []
                
                for b in range(batch_size):
                    # 计算该batch的概率分布
                    batch_probs = torch.softmax(top_logits[b], dim=-1)
                    cumulative_probs = torch.cumsum(batch_probs, dim=-1)
                    
                    # 保留累积概率小于top_p的tokens
                    mask = cumulative_probs <= top_p
                    # 确保至少保留一个token（第一个，概率最高的）
                    mask[0] = True
                    
                    batch_filtered_indices = top_indices[b][mask]
                    batch_filtered_logits = top_logits[b][mask]
                    
                    filtered_indices.append(batch_filtered_indices)
                    filtered_logits.append(batch_filtered_logits)
            else:
                # 如果没有 top-p 过滤，直接使用 top-k 结果
                filtered_indices = [top_indices[b] for b in range(batch_size)]
                filtered_logits = [top_logits[b] for b in range(batch_size)]
            
            next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for b in range(batch_size):
                if not unfinished[b]:
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue

                # 使用过滤后的候选 - 这是关键修复！
                current_candidates = filtered_indices[b]
                current_logits = filtered_logits[b]
                
                # 检查候选是否为空
                if len(current_candidates) == 0:
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue
                
                # 为每个候选token创建完整序列
                candidate_input_ids = []
                for t in range(len(current_candidates)):  # 使用正确的长度
                    candidate = torch.cat([
                        curr_input_ids[b:b+1],
                        current_candidates[t:t+1].unsqueeze(0)  # 使用过滤后的候选
                    ], dim=1)
                    candidate_input_ids.append(candidate)
                
                if not candidate_input_ids:  
                    continue
                    
                candidate_batch = torch.cat(candidate_input_ids, dim=0)
                candidate_texts = tokenizer.batch_decode(candidate_batch, skip_special_tokens=True)

                queries_responses = []
                for text in candidate_texts:
                    query = instructions.get_input(text)
                    response = instructions.get_response(text)
                    queries_responses.append((query, response))

                if hasattr(instructions, 'get_post'):
                    all_rewards = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
                else:
                    all_rewards = reward_models.get_reward_model_scores(queries_responses)

                # 计算加权 rewards
                weighted_rewards = torch.zeros(len(candidate_input_ids), device=device)
                for i in range(reward_models.num_rewards):
                    if isinstance(all_rewards[i], list) or isinstance(all_rewards[i], tuple):
                        reward_tensor = torch.tensor(all_rewards[i], device=device)
                    else:
                        reward_tensor = all_rewards[i]
                    if reward_tensor.dim() > 1:
                        reward_tensor = reward_tensor.squeeze()
                    
                    # 确保长度匹配
                    if len(reward_tensor) != len(candidate_input_ids):
                        print(f"Warning: Reward tensor length {len(reward_tensor)} doesn't match {len(candidate_input_ids)}")
                        # 截断或扩展到匹配长度
                        if len(reward_tensor) > len(candidate_input_ids):
                            reward_tensor = reward_tensor[:len(candidate_input_ids)]
                        else:
                            # 如果太短，用最后一个值填充
                            last_value = reward_tensor[-1] if len(reward_tensor) > 0 else 0.0
                            padding = torch.full((len(candidate_input_ids) - len(reward_tensor),), 
                                               last_value, device=device)
                            reward_tensor = torch.cat([reward_tensor, padding])
                    
                    weighted_rewards += preference_weights[i] * reward_tensor

                # 结合语言模型分数和奖励分数 - 使用正确的logits！
                combined_scores = current_logits[:len(candidate_input_ids)] + beta * weighted_rewards

                # 选择下一个token
                if generation_kwargs.get("do_sample", True):
                    sampling_probs = F.softmax(combined_scores / generation_kwargs.get("temperature", 1.0), dim=0)
                    next_token_idx = torch.multinomial(sampling_probs, num_samples=1)[0]
                else:
                    next_token_idx = torch.argmax(combined_scores)

                # 使用正确的候选索引 - 这也是关键修复！
                if next_token_idx < len(current_candidates):
                    next_tokens[b] = current_candidates[next_token_idx]
                else:
                    # fallback
                    next_tokens[b] = current_candidates[0]
                
                if step == 0 and b == 0:
                    print(f"DEBUG: Step {step}, batch {b}")
                    print(f"  Current candidates: {current_candidates}")
                    print(f"  Weighted rewards: {weighted_rewards}")
                    print(f"  Combined scores: {combined_scores}")
                    print(f"  Selected token: {next_tokens[b].item()}")

            # 更新序列
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

hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/sft/logs_trl/avocado/sft_famo_0.5')
    dpo_model_path: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/dpo/output/dev/famo_dpo/best_checkpoint')
    wandb_name: Optional[str] = field(default='evalnew_assistant_pretrained_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    num_samples: Optional[int] = field(default=100, metadata={"help": "number of samples to evaluate"})

    beta: Optional[float] = field(default=2, metadata={"help": "beta parameter for reward influence, paper used w=1.5 for LLaMA-7B"})
    topk: Optional[int] = field(default=10, metadata={"help": "topk parameter for candidate tokens, paper used k=10"})
    preference_weights: Optional[str] = field(default="0.5,0.5", metadata={"help": "comma-separated weights for reward models"})
    use_reward_guidance: Optional[bool] = field(default=True, metadata={"help": "whether to use reward-guided generation"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
dpo_model_path = script_args.dpo_model_path
tokenizer_name = script_args.base_model_name

model_path = dpo_model_path if (dpo_model_path is not None and os.path.exists(dpo_model_path)) else base_model_name
model_type = "DPO" if model_path == dpo_model_path else "SFT"
print(f"Using {model_type} model for evaluation: {model_path}")

process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
preference_weights = [float(x.strip()) for x in script_args.preference_weights.split(',')]
print('Reward models: ', reward_names)
print('Preference weights: ', preference_weights)

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
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) #, reward_stats_path) 
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)


set_seed(8888)
tokenizer = load_main_tokenizer(tokenizer_name)
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

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.95, 
    "do_sample": False,
}


### for evaluation
print('evaluation........')
tokenizer.padding_side = "left"

if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test', size=script_args.num_samples) 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

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

full_response_tensors = []
full_prompts = []

pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        # 根据参数决定是否使用reward引导生成
        if script_args.use_reward_guidance:
            print("Using reward-guided generation...")
            # 使用reward引导生成函数
            response_tensors = reward_guided_generate(
                accelerator.unwrap_model(model),
                reward_models,
                batch['input_ids'], 
                batch['attention_mask'],
                instructions,
                preference_weights=preference_weights,
                beta=script_args.beta,  # 基于论文的最佳参数 
                topk=script_args.topk,  # 基于论文的最佳参数
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

full_prompts = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_responses = get_clean_data(full_responses, full_prompts)
# Compute score
queries_responses = [
    (instructions.get_input(text),  instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)

### merge data
### error here may because of old version of transformers/accelerate/peft
all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)


if process_id == 0:
    evaluation_result = {
        'prompt': all_full_prompts,
        'response': all_full_responses,
    }
    for i in range(reward_models.num_rewards):
        evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
        print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

    if script_args.use_reward_guidance:
        filename = os.path.join(
            script_args.save_directory, 
            script_args.wandb_name,
            f'eval_data_reward_guided_beta{script_args.beta}_topk{script_args.topk}.csv'
        )
    else:
        filename = os.path.join(
            script_args.save_directory, 
            script_args.wandb_name,
            'eval_data.csv'
        )
        
    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(filename)
    print(f"Results saved to {filename}")