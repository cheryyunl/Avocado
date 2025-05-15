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
from peft import PeftModel, PeftConfig
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_main_tokenizer, check_lora_in_model_path

# 设置临时目录到空间充足的分区
import tempfile
os.environ['TMPDIR'] = '/scratch1/tmp'
os.environ['TEMP'] = '/scratch1/tmp'
os.environ['TMP'] = '/scratch1/tmp'
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/scratch1/huggingface_cache'
os.environ['HF_HOME'] = '/scratch1/huggingface_cache'

# 创建必要目录
os.makedirs('/scratch1/tmp', exist_ok=True)
os.makedirs('/scratch1/huggingface_cache', exist_ok=True)

# HelpSteer属性名称
HELPSTEER_ATTRIBUTES = [
    'helpsteer-helpfulness',
    'helpsteer-correctness',
    'helpsteer-coherence',
    'helpsteer-honesty',  
    'helpsteer-complexity'  
]

GUIDANCE_INDICES = [0, 1, 2, 4]
GUIDANCE_ATTRIBUTES = [HELPSTEER_ATTRIBUTES[i] for i in GUIDANCE_INDICES]

# 定义多组preference weights来评估
PREFERENCE_WEIGHTS_LIST = [
    [0, 0, 0.9, 0.1],    # 主要关注coherence, 少量complexity
]

def reward_guided_generate(
    model, 
    reward_model, 
    input_ids, 
    attention_mask, 
    tokenizer,
    preference_weights=None,
    beta=1.5,   # reward影响系数
    topk=10,    # 考虑的候选token数量
    **generation_kwargs
):
    """使用reward model实时引导生成的函数"""
    print(f"reward_guided_generate called with beta={beta}, topk={topk}")
    print(f"preference_weights={preference_weights}")
    
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # 如果没有提供权重，默认平均分配
    if preference_weights is None:
        preference_weights = [1.0 / len(GUIDANCE_INDICES)] * len(GUIDANCE_INDICES)
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
            
            top_logits, top_indices = torch.topk(logits, topk, dim=-1)
            
            # 应用top-p过滤（正确实现）- 修复版本！
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
                    mask = cumulative_probs <= top_p  # 改为<=
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
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue

                # 使用过滤后的候选 - 关键修复！
                current_candidates = filtered_indices[b]
                current_logits = filtered_logits[b]
                
                # 检查候选tokens是否为空
                if len(current_candidates) == 0:
                    next_tokens[b] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    continue
                
                # 为每个候选token生成完整的候选序列 - 使用过滤后的候选！
                candidate_input_ids = []
                for t in range(len(current_candidates)):  # 使用过滤后的长度
                    candidate = torch.cat([
                        curr_input_ids[b:b+1],
                        current_candidates[t:t+1].unsqueeze(0)  # 使用过滤后的候选
                    ], dim=1)
                    candidate_input_ids.append(candidate)
                
                if not candidate_input_ids:  
                    continue
                    
                candidate_batch = torch.cat(candidate_input_ids, dim=0)
                candidate_texts = tokenizer.batch_decode(candidate_batch, skip_special_tokens=True)

                # 提取原始prompt
                prompt_text = tokenizer.decode(input_ids[b], skip_special_tokens=True)
                prompt_text = prompt_text.replace("Human: ", "").replace("\nAssistant: ", "")
                
                # 提取每个候选的response部分
                query_response_pairs = []
                for text in candidate_texts:
                    if text.startswith(prompt_text):
                        response = text[len(prompt_text):]
                    else:
                        response = text
                    query_response_pairs.append((prompt_text, response.strip()))

                # 评估候选响应
                rewards_list = reward_model.get_reward_scores(query_response_pairs)

                # 计算加权reward
                weighted_rewards = torch.zeros(len(candidate_input_ids), device=device)
                for i, idx in enumerate(GUIDANCE_INDICES):
                    if idx < len(rewards_list) and rewards_list[idx]:
                        reward_tensor = torch.tensor(rewards_list[idx], device=device)
                        if reward_tensor.dim() > 1:
                            reward_tensor = reward_tensor.squeeze()
                        
                        # 确保长度匹配
                        if len(reward_tensor) != len(candidate_input_ids):
                            print(f"Warning: Reward tensor length {len(reward_tensor)} doesn't match {len(candidate_input_ids)}")
                            if len(reward_tensor) > len(candidate_input_ids):
                                reward_tensor = reward_tensor[:len(candidate_input_ids)]
                            else:
                                # 用最后一个值填充
                                last_value = reward_tensor[-1] if len(reward_tensor) > 0 else 0.0
                                padding = torch.full((len(candidate_input_ids) - len(reward_tensor),), 
                                                   last_value, device=device)
                                reward_tensor = torch.cat([reward_tensor, padding])
                        
                        weighted_rewards += preference_weights[i] * reward_tensor

                # 结合语言模型分数和reward分数 - 使用正确的logits！
                combined_scores = current_logits[:len(candidate_input_ids)] + beta * weighted_rewards

                # 根据生成参数决定是采样还是贪婪选择
                if generation_kwargs.get("do_sample", True):
                    sampling_probs = torch.softmax(combined_scores / generation_kwargs.get("temperature", 1.0), dim=0)
                    next_token_idx = torch.multinomial(sampling_probs, num_samples=1)[0]
                else:
                    next_token_idx = torch.argmax(combined_scores)

                # 使用正确的候选索引！
                if next_token_idx < len(current_candidates):
                    next_tokens[b] = current_candidates[next_token_idx]
                else:
                    next_tokens[b] = current_candidates[0]
                
                if step == 0 and b == 0:
                    print(f"DEBUG: Step {step}, batch {b}")
                    print(f"  Current candidates: {current_candidates}")
                    print(f"  Weighted rewards: {weighted_rewards}")
                    print(f"  Combined scores: {combined_scores}")
                    print(f"  Selected token: {next_tokens[b].item()}")

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

class HelpSteerRewardModel:
    def __init__(self, model_path="RLHFlow/RewardModel-Mistral-7B-for-DPA-v1", device="cuda"):
        self.device = device
        print(f"Loading HelpSteer reward model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 输入模板
        self.input_template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        self.num_rewards = len(HELPSTEER_ATTRIBUTES)
        print(f"Loaded HelpSteer model with {self.num_rewards} attributes")
        
    def get_reward_scores(self, query_response_pairs):
        """评估一批query-response对，返回每个属性的评分"""
        all_scores = [[] for _ in range(self.num_rewards)]
        
        for prompt, response in tqdm(query_response_pairs, desc="Evaluating responses"):
            # 使用模板格式化输入
            formatted_input = self.input_template.format(prompt=prompt, response=response)
            
            # 编码输入
            model_inputs = self.tokenizer(
                formatted_input, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                raw_scores = self.model(**model_inputs).logits.squeeze().cpu().float().numpy()
                
                helpsteer_scores = (raw_scores[:self.num_rewards] - 10) / 20
                
                for i in range(self.num_rewards):
                    all_scores[i].append(float(helpsteer_scores[i]))
                    
                if len(all_scores[0]) == 1:
                    print(f"Raw scores: {raw_scores[:self.num_rewards]}")
                    print(f"HelpSteer scores: {helpsteer_scores}")
        
        return all_scores

def build_helpsteer_eval_dataset(helpsteer_path, tokenizer, split='validation', seed=42, size=None):
    ds = load_dataset(helpsteer_path, split=split)
    ds = ds.shuffle(seed=seed)
    
    if size is not None:
        ds = ds.select(range(size))
    
    def tokenize(sample):
        sample['prompt'] = f"Human: {sample['prompt']}\nAssistant: "
        sample['original_prompt'] = sample['prompt'].replace("Human: ", "").replace("\nAssistant: ", "")
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        sample["attention_mask"] = [1] * len(sample["input_ids"])
        return sample
    
    ds_processed = ds.map(tokenize, batched=False, num_proc=20)
    
    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    
    columns_to_remove = ['helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity', 'response']
    columns_to_remove = [col for col in columns_to_remove if col in ds_processed.column_names]
    if columns_to_remove:
        ds_processed = ds_processed.remove_columns(columns_to_remove)
    ds_processed.set_format(type="torch")
    
    return ds_processed

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/sft/logs_trl/sft_hh_famo_helpsteer_mixed_model')  # 改为/scratch1
    base_model_name: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/dpo/model/famo-helpsteer-5000')
    dpo_model_path: Optional[str] = field(default=None)
    wandb_name: Optional[str] = field(default='eval_helpsteer_multiple_weights', metadata={"help": "Name for this experiment"})
    dataset_path: Optional[str] = field(default='nvidia/HelpSteer', metadata={"help": "Dataset to evaluate on"})
    reward_model_path: Optional[str] = field(default='RLHFlow/RewardModel-Mistral-7B-for-DPA-v1')
    num_samples: Optional[int] = field(default=0, metadata={"help": "Total number of samples to evaluate (0 for all)"})  # 改为0
    split: Optional[str] = field(default='validation', metadata={"help": "Dataset split to use"})
    
    # 新增参数，用于reward引导生成
    beta: Optional[float] = field(default=1.5, metadata={"help": "beta parameter for reward influence"})
    topk: Optional[int] = field(default=10, metadata={"help": "topk parameter for candidate tokens"})
    use_reward_guidance: Optional[bool] = field(default=True, metadata={"help": "whether to use reward-guided generation"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    base_model_name = script_args.base_model_name
    tokenizer_name = script_args.base_model_name
    dpo_model_path = script_args.dpo_model_path

    model_path = dpo_model_path if (dpo_model_path is not None and os.path.exists(dpo_model_path)) else base_model_name
    model_type = "DPO" if model_path == dpo_model_path else "SFT"
    print(f"Using {model_type} model for evaluation: {model_path}")
    
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
    tokenizer.padding_side = "left"  
    
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
    
    # 如果模型有merge_and_unload方法，则使用它
    if hasattr(model, 'merge_and_unload'):
        print("Merging and unloading adapters...")
        model = model.merge_and_unload()

    reward_model = HelpSteerRewardModel(script_args.reward_model_path, device=f"cuda:{gpu_id}")
    
    # 生成参数
    generation_kwargs = {
        "max_new_tokens": 128,
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.5, 
        "do_sample": False,
        "temperature": 0.3,
    }
    
    # 加载评估数据集（只需要加载一次）
    print('Loading evaluation dataset...')
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
    
    # 存储所有preference weights的结果
    all_results = {}
    
    # 循环评估每组preference weights
    for pref_idx, preference_weights in enumerate(PREFERENCE_WEIGHTS_LIST):
        print(f"\n{'='*50}")
        print(f"Evaluating preference weights: {preference_weights}")
        print(f"Run {pref_idx + 1} of {len(PREFERENCE_WEIGHTS_LIST)}")
        print(f"{'='*50}\n")
        
        # 评估循环
        full_response_tensors = []
        full_prompts = []
        original_prompts = []
        
        total_steps = len(valid_dataset) // valid_batch_size // num_processes + 1
        pbar = tqdm(total=total_steps, desc=f"GPU {gpu_id} generating with weights {preference_weights}")
        
        # 生成回复
        with torch.no_grad():
            for i, batch in enumerate(valid_data_loader):
                # 保存原始prompt
                if 'original_prompt' in batch:
                    original_prompts.extend(batch['original_prompt'])
                
                # 根据参数决定是否使用reward引导生成
                if script_args.use_reward_guidance:
                    if i == 0:  # 只在第一个批次打印
                        print(f"GPU {gpu_id} using reward-guided generation (beta={script_args.beta}, topk={script_args.topk})...")
                    
                    # 使用reward引导生成函数
                    response_tensors = reward_guided_generate(
                        accelerator.unwrap_model(model),
                        reward_model,
                        batch['input_ids'], 
                        batch['attention_mask'],
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
        
        # 解码生成的文本
        input_texts = tokenizer.batch_decode(full_prompts, skip_special_tokens=True)
        full_responses = tokenizer.batch_decode(full_response_tensors, skip_special_tokens=True)
        
        # 提取助手回复部分
        clean_responses = []
        for full_text, prompt in zip(full_responses, input_texts):
            if full_text.startswith(prompt):
                response = full_text[len(prompt):]
            else:
                if "Assistant:" in full_text:
                    response = full_text.split("Assistant:", 1)[1]
                else:
                    response = full_text
            
            clean_responses.append(response.strip())
        
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
                print(f'Average {attr_name} score: {np.mean(all_rewards[i]):.4f}')
            
            # 计算整体平均分
            all_scores_array = np.array([all_rewards[i] for i in range(len(HELPSTEER_ATTRIBUTES))])
            overall_scores = np.mean(all_scores_array, axis=0)
            evaluation_result['overall_score'] = overall_scores
            print(f'Overall average score: {np.mean(overall_scores):.4f}')
            
            # 创建包含preference weights信息的文件名
            weight_str = f"{preference_weights[0]}-{preference_weights[1]}-{preference_weights[2]}-{preference_weights[3]}"
            if script_args.use_reward_guidance:
                filename = os.path.join(
                    script_args.save_directory, 
                    script_args.wandb_name,
                    f'helpsteer_eval_reward_guided_beta{script_args.beta}_topk{script_args.topk}_weights{weight_str}.csv'
                )
            else:
                filename = os.path.join(
                    script_args.save_directory, 
                    script_args.wandb_name,
                    f'helpsteer_eval_weights{weight_str}.csv'
                )
            
            # 保存结果到CSV
            dataframe = pd.DataFrame(evaluation_result)
            dataframe.to_csv(filename)
            print(f"Results saved to {filename}")
            
            # 收集结果供最终汇总
            weight_key = weight_str
            all_results[weight_key] = {
                'weights': preference_weights,
                'scores': [np.mean(all_rewards[i]) for i in range(reward_model.num_rewards)],
                'overall_score': np.mean(overall_scores),
                'filename': filename
            }
        
        # 在每次循环后清理内存
        del full_response_tensors, full_prompts, full_responses, clean_responses
        del original_prompts, query_response_pairs, rewards_list
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print(f"Memory cleaned after run {pref_idx + 1}")
    
    # 在所有评估完成后汇总结果
    if process_id == 0:
        print(f"\n{'='*50}")
        print("FINAL SUMMARY")
        print(f"{'='*50}")
        print(f"{'Weights':<20} {'Helpfulness':<12} {'Correctness':<12} {'Coherence':<12} {'Complexity':<12} {'Overall':<10} {'File'}")
        print("-" * 100)
        
        for weight_key, result in all_results.items():
            weights = result['weights']
            scores = result['scores']
            overall = result['overall_score']
            filename = os.path.basename(result['filename'])
            
            # 只显示guidance attributes对应的分数
            guidance_scores = [scores[i] for i in GUIDANCE_INDICES]
            print(f"{weights:<20} {guidance_scores[0]:<12.4f} {guidance_scores[1]:<12.4f} {guidance_scores[2]:<12.4f} {guidance_scores[3]:<12.4f} {overall:<10.4f} {filename}")
        
        # 保存汇总结果
        summary_data = []
        for weight_key, result in all_results.items():
            weights = result['weights']
            scores = result['scores']
            guidance_scores = [scores[i] for i in GUIDANCE_INDICES]
            
            summary_data.append({
                'preference_weights': str(weights),
                'helpfulness_weight': weights[0],
                'correctness_weight': weights[1], 
                'coherence_weight': weights[2],
                'complexity_weight': weights[3],
                'helpfulness_score': guidance_scores[0],
                'correctness_score': guidance_scores[1],
                'coherence_score': guidance_scores[2],
                'complexity_score': guidance_scores[3],
                'overall_score': result['overall_score'],
                'filename': result['filename']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = os.path.join(
            script_args.save_directory, 
            script_args.wandb_name,
            f'helpsteer_preference_weights_summary_beta{script_args.beta}.csv'
        )
        summary_df.to_csv(summary_filename, index=False)
        print(f"\nSummary saved to {summary_filename}")

if __name__ == "__main__":
    main()