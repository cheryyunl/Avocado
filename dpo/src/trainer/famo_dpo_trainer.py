from trl import DPOTrainer
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from torch.utils.data import DataLoader, DistributedSampler

class TaskSpecificSampler(DistributedSampler):
    def __init__(self, dataset, n_tasks, **kwargs):
        from accelerate import Accelerator
        process_id = Accelerator().process_index
        num_gpus = Accelerator().num_processes
        assert num_gpus % n_tasks == 0
        gpus_per_task = num_gpus // n_tasks
        self.task_id = process_id // gpus_per_task
        local_rank = process_id % gpus_per_task
        local_num_replicas = gpus_per_task
        datasets = split_dataset(dataset)
        self.task_dataset = datasets[self.task_id]
        self.offset = sum(len(datasets[i]) for i in range(self.task_id))
        
        super().__init__(self.task_dataset,        
                        num_replicas=local_num_replicas,
                        rank=local_rank, **kwargs)

    def __iter__(self):
        indices = super().__iter__()
        return iter([idx + self.offset for idx in indices])
    
class FAMODPOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        args: TrainingArguments,
        beta: float,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
        peft_config=None,
        max_length=None,
        max_prompt_length=None,
        max_target_length=None,
        generate_during_eval=True,
        n_tasks=2,  # 添加任务数量参数
        task_weights=None,  # 可选的任务权重
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            beta=beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            generate_during_eval=generate_during_eval,
            **kwargs
        )
        self.n_tasks = n_tasks
        self.task_weights = task_weights if task_weights is not None else torch.ones(n_tasks) / n_tasks
        self._signature_columns = list(self._signature_columns) + ["task_id"]
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        
        # 使用多任务采样器
        self.sampler = TaskSpecificSampler(train_dataset, self.n_tasks)
        
        dataloader_params["sampler"] = self.sampler
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        
        return DataLoader(train_dataset, **dataloader_params)
    
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        if "task_id" in inputs:
            inputs["task_id"] = inputs["task_id"]
        return inputs
    
    
    def _tokenize_input_for_dpo(self, examples, is_train=True):
        tokenized_examples = super()._tokenize_input_for_dpo(examples, is_train)
        
        if "task_id" in examples:
            tokenized_examples["task_id"] = examples["task_id"]
            
        return tokenized_examples