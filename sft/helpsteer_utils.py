from datasets import load_dataset, concatenate_datasets
import torch
import os

def build_helpsteer_positive_dataset(helpsteer_path, tokenizer, split='train', seed=42):
    ds_helpfulness = load_dataset(helpsteer_path, data_dir="helpfulness-positive", split=split)
    ds_correctness = load_dataset(helpsteer_path, data_dir="correctness-positive", split=split)
    ds_coherence = load_dataset(helpsteer_path, data_dir="coherence-positive", split=split)
    ds_complexity = load_dataset(helpsteer_path, data_dir="complexity-positive", split=split)

    ds_helpfulness = ds_helpfulness.map(lambda x: {"task_id": 0})
    ds_correctness = ds_correctness.map(lambda x: {"task_id": 1})
    ds_coherence = ds_coherence.map(lambda x: {"task_id": 2})
    ds_complexity = ds_complexity.map(lambda x: {"task_id": 3})
    
    ds_helpfulness = ds_helpfulness.shuffle(seed=seed)
    ds_correctness = ds_correctness.shuffle(seed=seed)
    ds_coherence = ds_coherence.shuffle(seed=seed)
    ds_complexity = ds_complexity.shuffle(seed=seed)
    
    ds = concatenate_datasets([ds_helpfulness, ds_correctness, ds_coherence, ds_complexity])
    
    def tokenize(sample):
        sample['text'] = sample['prompt'] + sample['response']
        prompt_ids = tokenizer.encode(sample['prompt'])
        response_ids = tokenizer.encode(sample['response']) + [tokenizer.eos_token_id]
        
        sample["input_ids"] = prompt_ids + response_ids
        sample["labels"] = [-100] * len(prompt_ids) + response_ids
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample
    ds_processed = ds.map(tokenize, batched=False, num_proc=30)

    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    ds_processed = ds_processed.remove_columns([
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',  
    'prompt', 'response'  
    ])

    ds_processed.set_format(type="torch")
    
    return ds_processed

def build_helpsteer_negative_dataset(helpsteer_path, tokenizer, split='train', seed=42):
    ds_helpfulness = load_dataset(helpsteer_path, data_dir="helpfulness-negative", split=split)
    ds_correctness = load_dataset(helpsteer_path, data_dir="correctness-negative", split=split)
    ds_coherence = load_dataset(helpsteer_path, data_dir="coherence-negative", split=split)
    ds_complexity = load_dataset(helpsteer_path, data_dir="complexity-negative", split=split)
    
    ds_helpfulness = ds_helpfulness.map(lambda x: {"task_id": 0})
    ds_correctness = ds_correctness.map(lambda x: {"task_id": 1})
    ds_coherence = ds_coherence.map(lambda x: {"task_id": 2})
    ds_complexity = ds_complexity.map(lambda x: {"task_id": 3})
    
    ds_helpfulness = ds_helpfulness.shuffle(seed=seed)
    ds_correctness = ds_correctness.shuffle(seed=seed)
    ds_coherence = ds_coherence.shuffle(seed=seed)
    ds_complexity = ds_complexity.shuffle(seed=seed)
    
    ds = concatenate_datasets([ds_helpfulness, ds_correctness, ds_coherence, ds_complexity])
    
    def tokenize(sample):
        sample['text'] = sample['prompt'] + sample['response']
        prompt_ids = tokenizer.encode(sample['prompt'])
        response_ids = tokenizer.encode(sample['response']) + [tokenizer.eos_token_id]
        
        sample["input_ids"] = prompt_ids + response_ids
        sample["labels"] = [-100] * len(prompt_ids) + response_ids
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample
    
    ds_processed = ds.map(tokenize, batched=False, num_proc=30)

    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    ds_processed = ds_processed.remove_columns([
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',  
    'prompt', 'response'  
    ])

    ds_processed.set_format(type="torch")
    
    return ds_processed




