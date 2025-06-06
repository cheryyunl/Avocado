from datasets import load_dataset, concatenate_datasets
import torch
import os
from transformers import DataCollatorForLanguageModeling
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


def build_helpsteer_mixed_dataset(helpsteer_path, tokenizer, split='train', seed=42):
    ds_helpfulness = load_dataset(helpsteer_path, data_dir="helpfulness-positive", split=split)
    ds_correctness = load_dataset(helpsteer_path, data_dir="correctness-positive", split=split)
    ds_coherence = load_dataset(helpsteer_path, data_dir="coherence-positive", split=split)
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

def build_helpsteer_dataset(helpsteer_path, tokenizer, split='train', seed=42):
    ds = load_dataset(helpsteer_path, split=split)
    
    def tokenize(sample):
        prompt_ids = tokenizer.encode(sample['prompt'])
        response_ids = tokenizer.encode(sample['response']) + [tokenizer.eos_token_id]
        sample["input_ids"] = prompt_ids + response_ids
        sample["labels"] = [-100] * len(prompt_ids) + response_ids  
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    
    ds_processed = ds.map(tokenize, batched=False, num_proc=30)

    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    ds_processed = ds_processed.remove_columns([
    'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity',  
    'prompt', 'response'  
    ])

    ds_processed.set_format(type="torch")
    
    return ds_processed

def build_helpsteer_eval_dataset(helpsteer_path, tokenizer, split='validation', seed=42):
    ds_helpfulness = load_dataset(helpsteer_path, data_dir="helpfulness-positive", split=split)
    ds_correctness = load_dataset(helpsteer_path, data_dir="correctness-positive", split=split)
    ds_coherence = load_dataset(helpsteer_path, data_dir="coherence-positive", split=split)
    ds_complexity = load_dataset(helpsteer_path, data_dir="complexity-negative", split=split)

    ds = concatenate_datasets([ds_helpfulness, ds_correctness, ds_coherence, ds_complexity])
    ds = ds.shuffle(seed=seed)
    
    def tokenize_for_eval(sample):
        formatted_prompt = f"Human: {sample['prompt']}\nAssistant: "
        prompt_ids = tokenizer.encode(formatted_prompt, truncation=True, max_length=1024)
 
        sample["input_ids"] = prompt_ids
        sample["attention_mask"] = [1] * len(prompt_ids)
        sample["original_prompt"] = sample["prompt"]  # 保存原始prompt用于评估
        return sample

    ds_processed = ds.map(tokenize_for_eval, remove_columns=[
        'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity', 'response'
    ])

    ds_processed = ds_processed.filter(lambda x: len(x["input_ids"]) <= 1024 and len(x["input_ids"]) >= 8)
    
    return ds_processed

class HSDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        task_ids = None
        if all('task_id' in example for example in examples):
            task_ids = []
            for example in examples:
                tid = example.pop('task_id')
                if isinstance(tid, torch.Tensor):
                    tid = tid.item() if tid.numel() == 1 else tid[0].item()
                task_ids.append(tid)
        
        for example in examples:
            if "labels" in example and isinstance(example["labels"], list):
                if any(isinstance(x, list) for x in example["labels"]):
                    example["labels"] = [x[0] if isinstance(x, list) and x else x for x in example["labels"]]

        batch = super().torch_call(examples)
        if task_ids is not None:
            batch['task_id'] = torch.tensor(task_ids, dtype=torch.long)
        return batch