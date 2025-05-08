import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict
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
tqdm.pandas()

# 定义HelpSteer的属性名称 (去掉verbosity)
HELPSTEER_ATTRIBUTES = [
    'helpsteer-helpfulness', 
    'helpsteer-correctness', 
    'helpsteer-coherence',
    'helpsteer-complexity'
]

# 属性到数据集子集的映射
ATTRIBUTE_SUBSETS = {
    'helpsteer-helpfulness': 'helpfulness-positive',
    'helpsteer-correctness': 'correctness-positive',
    'helpsteer-coherence': 'coherence-positive',
    'helpsteer-complexity': 'complexity-negative'
}

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
                # 只获取我们需要的四个属性评分
                for i in range(self.num_rewards):
                    all_scores[i].append(output.score[i].cpu().float().item())
        
        return all_scores

def build_helpsteer_eval_dataset(helpsteer_path, tokenizer, split='test', seed=42):
    """加载HelpSteer数据集，但只保留prompts用于评估"""
    datasets = {}
    
    # 加载每个属性对应的子集
    for attr_name, subset_name in ATTRIBUTE_SUBSETS.items():
        try:
            dataset = load_dataset(helpsteer_path, data_dir=subset_name, split=split)
            datasets[attr_name] = dataset
            print(f"Loaded {len(dataset)} samples for {attr_name}")
        except Exception as e:
            print(f"Error loading {attr_name} dataset: {e}")
    
    # 合并所有子集
    all_datasets = []
    for attr_name, dataset in datasets.items():
        dataset = dataset.add_column("attribute", [attr_name] * len(dataset))
        all_datasets.append(dataset)
    
    combined_dataset = concatenate_datasets(all_datasets)
    combined_dataset = combined_dataset.shuffle(seed=seed)
    
    def preprocess_function(examples):
        prompts = examples["prompt"]
        attributes = examples["attribute"]
        texts = []
        
        for prompt in prompts:
            # 只保留prompt部分用于生成
            text = f"Human: {prompt}\nAssistant: "
            texts.append(text)
        
        tokenized = tokenizer(texts, padding=False, truncation=True)
        
        # 添加原始prompt和属性用于后续评估
        tokenized["original_prompt"] = prompts
        tokenized["attribute"] = attributes
        return tokenized
    
    tokenized_dataset = combined_dataset.map(preprocess_function, batched=True)
    
    # 过滤太长的样本
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= 1024)
    
    return tokenized_dataset

def get_clean_response(full_text, prompt):
    """从生成的文本中提取助手的回复"""
    # 移除prompt部分
    if full_text.startswith(prompt):
        response = full_text[len(prompt):]
    else:
        # 尝试不同的分割方法
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
    dataset_path: Optional[str] = field(default='HelpSteer/help-steer', metadata={"help": "Dataset to evaluate on"})
    reward_model_path: Optional[str] = field(default='nicolinho/QRM-Llama3.1-8B-v2')
    num_samples: Optional[int] = field(default=100, metadata={"help": "Number of samples to evaluate per attribute"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    base_model_name = script_args.base_model_name
    tokenizer_name = script_args.base_model_name
    print('base model: ', base_model_name)
    
    process_id = Accelerator().local_process_index 
    gpu_id = process_id 
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
    
    # 初始化HelpSteer reward model
    reward_model = HelpSteerRewardModel(script_args.reward_model_path, device=f"cuda:{gpu_id}")
    os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)
    
    set_seed(8888)
    tokenizer = load_main_tokenizer(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id, 
    )
    model.resize_token_embeddings(len(tokenizer))
    if check_lora_in_model_path(model, base_model_name):
        model = PeftModel.from_pretrained(model, base_model_name)
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    
    generation_kwargs = {
        "max_new_tokens": 128, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }
    
    print('Loading evaluation dataset...')
    tokenizer.padding_side = "left"
    
    valid_dataset = build_helpsteer_eval_dataset(
        script_args.dataset_path, 
        tokenizer, 
        split='validation'
    )

    if script_args.num_samples > 0:
        attr_datasets = []
        for attr in ATTRIBUTE_SUBSETS.keys():
            attr_subset = valid_dataset.filter(lambda x: x["attribute"] == attr)
            if len(attr_subset) > script_args.num_samples:
                attr_subset = attr_subset.select(range(script_args.num_samples))
            attr_datasets.append(attr_subset)
        
        valid_dataset = concatenate_datasets(attr_datasets)
        valid_dataset = valid_dataset.shuffle(seed=8888)
    
    print(f"Size of the validation set: {len(valid_dataset)}")
    
    valid_batch_size = 1  
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=valid_batch_size, 
        drop_last=False, 
        collate_fn=data_collator
    )
    
    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    full_response_tensors = []
    full_prompts = []
    original_prompts = []
    sample_attributes = []
    
    pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes + 1)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            if 'original_prompt' in batch:
                original_prompts.extend(batch['original_prompt'])
            if 'attribute' in batch:
                sample_attributes.extend(batch['attribute'])
            
            response_tensors = accelerator.unwrap_model(model).generate(
                batch['input_ids'], 
                attention_mask=batch['attention_mask'], 
                **generation_kwargs
            )
            
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    input_texts = tokenizer.batch_decode(full_prompts, skip_special_tokens=True)
    full_responses = tokenizer.batch_decode(full_response_tensors, skip_special_tokens=True)

    clean_responses = []
    for i, (full_text, prompt) in enumerate(zip(full_responses, input_texts)):
        response = get_clean_response(full_text, prompt)
        clean_responses.append(response)
    
    query_response_pairs = list(zip(original_prompts, clean_responses))
    rewards_list = reward_model.get_reward_scores(query_response_pairs)
    all_rewards = []
    for i in range(reward_model.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    
    all_full_prompts = accelerator.gather_for_metrics(original_prompts)
    all_full_responses = accelerator.gather_for_metrics(clean_responses)
    all_attributes = accelerator.gather_for_metrics(sample_attributes)
    
    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
            'source_attribute': all_attributes
        }

        for i, attr_name in enumerate(HELPSTEER_ATTRIBUTES):
            evaluation_result[attr_name] = all_rewards[i]
            print(f'Average {attr_name} score: {np.mean(all_rewards[i])}')
        
        all_scores_array = np.array([all_rewards[i] for i in range(len(HELPSTEER_ATTRIBUTES))])
        overall_scores = np.mean(all_scores_array, axis=0)
        evaluation_result['overall_score'] = overall_scores
        print(f'Overall average score: {np.mean(overall_scores)}')

        for attr in ATTRIBUTE_SUBSETS.keys():
            mask = np.array([a == attr for a in all_attributes])
            if np.any(mask):
                for i, attr_name in enumerate(HELPSTEER_ATTRIBUTES):
                    attr_scores = np.array(all_rewards[i])[mask]
                    print(f'Average {attr_name} score for {attr} samples: {np.mean(attr_scores)}')
        
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