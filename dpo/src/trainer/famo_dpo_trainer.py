from src.trainer.dpo_trainer import DPOTrainer, DPODataMapFunc, DPODataCollatorWithPadding
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import TrainingArguments, PreTrainedTokenizerBase, is_datasets_available
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, DistributedSampler
from src.utils import split_dataset, common_prefix_length, pad_labels
from dataclasses import dataclass
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import wandb

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

@dataclass
class DPODataMapFunc:
    """Map raw texts to tokens, attention masks, and labels."""
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    completion_only: Optional[bool] = True

    def __call__(self, examples):
        new_examples = {
            "prompt_chosen_input_ids": [],
            "prompt_chosen_attention_mask": [],
            "prompt_chosen_labels": [],

            "prompt_rejected_input_ids": [],
            "prompt_rejected_attention_mask": [],
            "prompt_rejected_labels": [],

            "prompt_input_ids": [],
            "prompt_attention_mask": [],

            "prompt": [],
            "task_id": [],  # 添加task_id字段
        }

        for i, (prompt, chosen, rejected) in enumerate(zip(examples["prompt"], examples["chosen"], examples["rejected"])):
            prompt_tokens = self.tokenizer(prompt)
            prompt_chosen_tokens = self.tokenizer(prompt + chosen)
            prompt_rejected_tokens = self.tokenizer(prompt + rejected)
            # add EOS to response
            prompt_chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            prompt_chosen_tokens["attention_mask"].append(1)
            prompt_rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            prompt_rejected_tokens["attention_mask"].append(1)

            prompt_len = common_prefix_length(prompt_chosen_tokens["input_ids"], prompt_rejected_tokens["input_ids"])

            for k, toks in {
                "prompt": prompt_tokens,
                "prompt_chosen": prompt_chosen_tokens,
                "prompt_rejected": prompt_rejected_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    new_examples[f"{k}_{type_key}"].append(tokens)
            
            for k, toks in {
                "prompt_chosen": prompt_chosen_tokens,
                "prompt_rejected": prompt_rejected_tokens,
            }.items():
                labels = toks["input_ids"].copy()
                if self.completion_only:
                    labels[:prompt_len] = [self.label_pad_token_id] * prompt_len
                new_examples[f"{k}_labels"].append(labels) 

            # 添加task_id
            if "task_id" in examples:
                new_examples["task_id"].append(examples["task_id"][i])

        new_examples["prompt"] = examples["prompt"]

        return new_examples


@dataclass
class DPODataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: Optional[int] = -100
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]], generate: Optional[bool] = False) -> Dict[str, Any]:
        """
        if not generate:
            batch = {
                "input_ids": ...,
                "attention_mask": ...,
                "labels": ...,
                "task_id": ...  # 新增
            }
        else:
            batch = {
                "prompt": ...,
                "prompt_input_ids": ...,
                "prompt_attention_mask": ...,
                "task_id": ...  # 新增
            }
        """
        if not generate:
            # 收集所有task_id
            task_ids = []
            if all("task_id" in feature for feature in features):
                task_ids = [feature["task_id"] for feature in features]
            
            # `chosen` and `rejected` merged into a single batch for more efficient batched forward pass;
            right_padding_features = []
            for feature in features:
                right_padding_features.append(
                    {
                        "input_ids": feature["prompt_chosen_input_ids"],
                        "attention_mask": feature["prompt_chosen_attention_mask"],
                        "labels": feature["prompt_chosen_labels"],
                    }
                )
            for feature in features:
                right_padding_features.append(
                    {
                        "input_ids": feature["prompt_rejected_input_ids"],
                        "attention_mask": feature["prompt_rejected_attention_mask"],
                        "labels": feature["prompt_rejected_labels"],
                    }
                )

            pad_labels(right_padding_features, self.tokenizer, self.pad_to_multiple_of, self.label_pad_token_id)

            right_padding_batch = self.tokenizer.pad(
                right_padding_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # 添加task_id到批次中
            if task_ids:
                #  需要复制每个task_id两次，因为每个样本既有chosen又有rejected
                # [id1, id2, id3] -> [id1, id2, id3, id1, id2, id3]
                duplicated_task_ids = task_ids + task_ids
                right_padding_batch["task_id"] = torch.tensor(duplicated_task_ids, dtype=torch.long)

            return right_padding_batch

        else:
            # 收集所有task_id
            task_ids = []
            if all("task_id" in feature for feature in features):
                task_ids = [feature["task_id"] for feature in features]

            left_padding_features = [] 
            padding_side_default = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            for feature in features:
                left_padding_features.append(
                    {
                        "input_ids": feature["prompt_input_ids"],
                        "attention_mask": feature["prompt_attention_mask"],
                    }
                )
            left_padding_batch = self.tokenizer.pad(
                left_padding_features,
                padding=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = padding_side_default

            result = {
                "prompt": [feature["prompt"] for feature in features],
                "prompt_input_ids": left_padding_batch["input_ids"],
                "prompt_attention_mask": left_padding_batch["attention_mask"],
            }
            
            if task_ids:
                result["task_id"] = torch.tensor(task_ids, dtype=torch.long)
            
            return result
    
class FAMODPOTrainer(DPOTrainer):
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        peft_config=None,
        packing=False,
        data_collator=None,
        tokenize_map_func=None,
        tokenizer=None,
        n_tasks: int=2,
        gamma: float = 0.01,
        w_lr: float = 0.01,
        famo_update_frequency=5,
        ema_alpha = None,
        init_steps = None,
        **trainer_args
    ):

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            data_collator=data_collator,
            tokenize_map_func=tokenize_map_func,
            tokenizer=tokenizer,
            **trainer_args
        )
        self.step_count = 0
        self.famo_update_frequency = famo_update_frequency
        self._signature_columns = ["input_ids", "labels", "attention_mask", "task_id"]
        self.compute_loss_func = None
        self.model_accepts_loss_kwargs = False
        self.args = args
        self.args.average_tokens_across_devices = True
        self.n_tasks = n_tasks
        self.ema_alpha = ema_alpha
        self.init_steps = init_steps

        self.min_losses = torch.zeros(n_tasks, device='cuda')
        self.w = torch.full((n_tasks,), 1.0 / n_tasks, requires_grad=True, device='cuda')
        if self.args.local_rank == 0:
            self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory
        }

        self.sampler = TaskSpecificSampler(train_dataset, self.n_tasks)

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self.sampler 
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    def eval_batch_metrics(self, model, inputs):
        with torch.no_grad():
            return self.get_batch_metrics(model, inputs, train_eval="train")
    
    def update_famo_weights(self, prev_loss, curr_loss):
        if not self.args.local_rank == 0:
            return
            
        delta = torch.zeros_like(prev_loss)

        if not hasattr(self, 'initial_losses') and self.step_count < self.init_steps:
            self.loss_ema = getattr(self, 'loss_ema', curr_loss.clone())
            self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * curr_loss
        elif not hasattr(self, 'initial_losses') and self.step_count == self.init_steps:
            self.initial_losses = self.loss_ema.clone()
    
        if hasattr(self, 'initial_losses'):
            norm_prev = prev_loss / torch.clamp(self.initial_losses, min=1e-5)
            norm_curr = curr_loss / torch.clamp(self.initial_losses, min=1e-5)
        else:
            norm_prev = prev_loss
            norm_curr = curr_loss

        delta = (norm_prev - self.min_losses + 1e-8).log() - \
                (norm_curr - self.min_losses + 1e-8).log()
        
        if torch.isnan(delta).any() or torch.isinf(delta).any():
            return
            
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                  self.w,
                                  grad_outputs=delta.detach())[0]
            
        self.w_opt.zero_grad()
        self.w.grad = d.to(self.w.device)
        self.w_opt.step()

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        famo_w = F.softmax(self.w, -1)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        task_ids = inputs.pop("task_id")
        task_id = task_ids[0].item()
        
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            orig_loss = self.compute_loss(model, inputs)
            loss = orig_loss * famo_w[task_id]

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        all_losses = torch.zeros(self.n_tasks, device=loss.device)
        all_losses[task_id] = orig_loss.detach()
        all_losses = self.accelerator.reduce(all_losses, "sum")
        
        if self.step_count > 0 and self.step_count % self.famo_update_frequency == 0:
            prev_task_id = self.prev_inputs.pop("task_id")[0].item()
            with torch.no_grad():
                new_loss, _ = self.eval_batch_metrics(model, self.prev_inputs)
            new_all_losses = torch.zeros(self.n_tasks, device=new_loss.device)
            new_all_losses[prev_task_id] = new_loss
            new_all_losses = self.accelerator.reduce(new_all_losses, "sum")
            
            if self.accelerator.is_main_process:
                self.update_famo_weights(self.prev_loss, new_all_losses)
                wandb.log({
                    **{f'task_{task_id}_weight': famo_w[task_id].item() for task_id in range(self.n_tasks)},
                    **{f'task_{task_id}_loss': all_losses[task_id].item() for task_id in range(self.n_tasks)},
                })

            self.accelerator.wait_for_everyone()
            torch.distributed.broadcast(self.w.detach(), src=0)
        

        inputs["task_id"] = task_ids
        self.prev_inputs = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
        self.prev_loss = all_losses.clone()
            
        self.step_count += 1

        del inputs

        return loss.detach()
