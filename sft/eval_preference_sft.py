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
from peft import PeftModel
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, check_lora_in_model_path, Instructions, Instructions_summary, \
                    build_dataset_eval, build_dataset_summary_eval, get_clean_data
tqdm.pandas()

# 新增函数：reward引导生成
def reward_guided_generate(
    model, 
    reward_models, 
    input_ids, 
    attention_mask, 
    instructions,
    preference_weights=None,
    beta=0.5,   # reward影响系数
    topk=10,    # 考虑的候选token数量，基于论文使用k=10
    **generation_kwargs
):
    """
    使用多个reward models实时引导生成的函数
    """
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 如果没有提供权重，默认平均分配
    if preference_weights is None:
        preference_weights = [1.0 / reward_models.num_rewards] * reward_models.num_rewards
    else:
        # 归一化权重
        total = sum(preference_weights)
        preference_weights = [w / total for w in preference_weights]

    # 开始token作为当前序列
    curr_input_ids = input_ids.clone()
    curr_attention_mask = attention_mask.clone()
    
    # 跟踪哪些序列已经完成生成
    unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # 最大生成token数量
    max_length = curr_input_ids.size(1) + generation_kwargs.get("max_new_tokens", 128)
    
    # 跟踪生成过程
    cached_output = None
    
    for _ in range(max_length - curr_input_ids.size(1)):
        # 如果所有序列都已完成，则停止
        if not unfinished.any():
            break
        
        # 使用模型获取下一个token的预测
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
            
            # 获取当前时间步骤的logits
            logits = model_outputs.logits[:, -1, :]
            
            # 应用温度参数
            if "temperature" in generation_kwargs and generation_kwargs["temperature"] > 0:
                logits = logits / generation_kwargs["temperature"]
            
            # 获取topk的token和它们的概率
            top_logits, top_indices = torch.topk(logits, topk, dim=-1)
            
            # 如果使用top_p采样，进一步筛选tokens
            if "top_p" in generation_kwargs and generation_kwargs["top_p"] < 1.0:
                top_p = generation_kwargs["top_p"]
                cumulative_probs = torch.softmax(top_logits, dim=-1)
                cumulative_probs = torch.cumsum(cumulative_probs, dim=-1)
                # 移除概率累积超过top_p的tokens
                mask = cumulative_probs < top_p
                mask = torch.cat([torch.ones_like(mask[:, :1], dtype=torch.bool), mask[:, :-1]], dim=1)
                top_indices = top_indices.masked_select(mask).view(batch_size, -1)
                top_logits = top_logits.masked_select(mask).view(batch_size, -1)
                # 重新计算topk，因为可能被top_p筛选减少了
                topk = min(topk, top_indices.size(1))
            
            top_probs = torch.softmax(top_logits, dim=-1)
            
            # 对于每个批次项计算reward
            next_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for b in range(batch_size):
                if not unfinished[b]:
                    continue
                    
                # 为每个候选token创建一个完整序列
                candidate_input_ids = []
                for t in range(topk):
                    if t < top_indices.size(1):  # 确保索引有效
                        candidate = torch.cat([
                            curr_input_ids[b:b+1],
                            top_indices[b:b+1, t:t+1]
                        ], dim=1)
                        candidate_input_ids.append(candidate)
                
                if not candidate_input_ids:  # 如果没有有效候选，继续下一个batch
                    continue
                    
                # 将所有候选项合并成一个大批次
                candidate_batch = torch.cat(candidate_input_ids, dim=0)
                
                # 解码候选序列
                candidate_texts = tokenizer.batch_decode(candidate_batch, skip_special_tokens=True)
                
                # 准备query-response对供reward模型评估
                queries_responses = []
                for text in candidate_texts:
                    query = instructions.get_input(text)
                    response = instructions.get_response(text)
                    queries_responses.append((query, response))
                
                # 获取所有reward模型的分数
                if hasattr(instructions, 'get_post'):
                    all_rewards = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
                else:
                    all_rewards = reward_models.get_reward_model_scores(queries_responses)
                
                # 计算加权reward
                weighted_rewards = torch.zeros(len(candidate_input_ids), device=device)
                for i in range(reward_models.num_rewards):
                    weighted_rewards += preference_weights[i] * all_rewards[i]
                
                # 使用reward调整token概率
                adjusted_probs = top_probs[b, :len(candidate_input_ids)] * torch.exp(beta * weighted_rewards)
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                
                # 根据调整后的概率采样下一个token
                if generation_kwargs.get("do_sample", False):
                    next_token_idx = torch.multinomial(adjusted_probs, num_samples=1)[0]
                else:
                    # 贪婪搜索，选择最高概率
                    next_token_idx = torch.argmax(adjusted_probs)
                
                if next_token_idx < top_indices.size(1):
                    next_tokens[b] = top_indices[b, next_token_idx]
            
            # 添加新token到序列中
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

# 原有代码继续...
# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='./huggingface_models/Llama-2-7b-hf')
    wandb_name: Optional[str] = field(default='evalnew_assistant_pretrained_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    
    # 新增参数用于reward引导生成
    beta: Optional[float] = field(default=1.5, metadata={"help": "beta parameter for reward influence, paper used w=1.5 for LLaMA-7B"})
    topk: Optional[int] = field(default=10, metadata={"help": "topk parameter for candidate tokens, paper used k=10"})
    preference_weights: Optional[str] = field(default="0.5,0.5", metadata={"help": "comma-separated weights for reward models"})
    use_reward_guidance: Optional[bool] = field(default=True, metadata={"help": "whether to use reward-guided generation"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
tokenizer_name = script_args.base_model_name
print('base model: ', base_model_name)

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
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16,  # faster inference than 8bit
    device_map=gpu_id, 
)
############# very important for padding
model.resize_token_embeddings(len(tokenizer))
if check_lora_in_model_path(model, base_model_name):
    model = PeftModel.from_pretrained(model, base_model_name)
if hasattr(model, 'merge_and_unload'):
    model = model.merge_and_unload()

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9, 
    "do_sample": True,
}


### for evaluation
print('evaluation........')
tokenizer.padding_side = "left"

if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
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