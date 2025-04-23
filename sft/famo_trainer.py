import torch
import torch.nn.functional as F
from trl import SFTTrainer
from torch import nn
from typing import Callable, Dict, List, Union, Any, Optional
from transformers import TrainingArguments
import datasets
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, DistributedSampler
from transformers.utils import is_sagemaker_mp_enabled, is_datasets_available, is_torch_mlu_available, is_torch_mps_available, is_torch_musa_available, is_torch_npu_available, is_torch_xpu_available
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import seed_worker
from multi_task_utils import split_dataset
import warnings
import sys
import time
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


class FAMOSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset,
        peft_config,
        packing,
        dataset_text_field,
        data_collator,
        n_tasks: int=2,
        gamma: float = 0.01,
        w_lr: float = 0.01,
        famo_update_frequency=5,
        rejected_ids = None,
        **trainer_args
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            data_collator=data_collator,
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
        self.rejected_ids = rejected_ids
        self.loss_scale = 1
        
        self.min_losses = torch.zeros(n_tasks, device='cuda')
        self.w = torch.full((n_tasks,), 1.0 / n_tasks, requires_grad=True, device='cuda')
        if self.args.local_rank == 0:
            self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)

    def update_famo_weights(self, prev_loss, curr_loss):
        if not self.args.local_rank == 0:
            return
            
        delta = torch.zeros_like(prev_loss)
        prev_loss_adjusted = prev_loss.clone()
        curr_loss_adjusted = curr_loss.clone()
                    
        if hasattr(self, 'rejected_ids') and self.rejected_ids is not None and len(self.rejected_ids) > 0:
            C = 5
            mask = torch.zeros_like(prev_loss_adjusted, dtype=torch.bool)
            mask[self.rejected_ids] = True
            prev_loss_adjusted[mask] = torch.max(prev_loss_adjusted[mask], torch.tensor(-C).to(prev_loss_adjusted.device))
            curr_loss_adjusted[mask] = torch.max(curr_loss_adjusted[mask], torch.tensor(-C).to(curr_loss_adjusted.device))
            prev_loss_adjusted[mask] += C
            curr_loss_adjusted[mask] += C

        # with torch.no_grad():
            # self.min_losses = torch.minimum(self.min_losses, curr_loss_adjusted)
            
        delta = (prev_loss_adjusted - self.min_losses + 1e-8).log() - \
                (curr_loss_adjusted - self.min_losses + 1e-8).log()
        
        if torch.isnan(delta).any() or torch.isinf(delta).any():
            return
            
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                  self.w,
                                  grad_outputs=delta.detach())[0]
            
        self.w_opt.zero_grad()
        self.w.grad = d.to(self.w.device)
        self.w_opt.step()


    def _prepare_non_packed_dataloader(
            self,
            processing_class,
            dataset,
            dataset_text_field: str,
            max_seq_length,
            formatting_func: Optional[Callable] = None,
            add_special_tokens=True,
            remove_unused_columns=True,
        ):
            def tokenize(element):
                outputs = processing_class(
                    element[dataset_text_field] if formatting_func is None else formatting_func(element),
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_overflowing_tokens=False,
                    return_length=False,
                )

                if formatting_func is not None and not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                
                outputs['task_id'] = element['task_id']

                return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "task_id": outputs["task_id"]}

            signature_columns = ["input_ids", "labels", "attention_mask", "task_id"]

            if dataset.column_names is not None:  # None for IterableDataset
                extra_columns = list(set(dataset.column_names) - set(signature_columns))
            else:
                extra_columns = []

            if not remove_unused_columns and len(extra_columns) > 0:
                warnings.warn(
                    "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                    f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
                )
            map_kwargs = {
                "batched": True,
                "remove_columns": dataset.column_names if remove_unused_columns else None,
                "batch_size": self.dataset_batch_size,
            }
            if isinstance(dataset, datasets.Dataset):
                map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
            tokenized_dataset = dataset.map(tokenize, **map_kwargs)

            return tokenized_dataset

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
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
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        self.sampler = TaskSpecificSampler(train_dataset, self.n_tasks)

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self.sampler 
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def eval_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        with torch.no_grad():
            outputs = model(**inputs)
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                unwrapped_model = self.accelerator.unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                # User-defined compute_loss function
                if self.compute_loss_func is not None:
                    loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
                elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
                loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
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
            if self.rejected_ids is not None and task_id in self.rejected_ids:
                orig_loss = - self.loss_scale * torch.log(1 + orig_loss)
                # beta = 0.1
                # orig_loss = -F.logsigmoid(beta * orig_loss).mean() * 2 / beta 
            loss = orig_loss * famo_w[task_id]
        
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if num_items_in_batch is None:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)

        all_losses = torch.zeros(self.n_tasks, device=loss.device)
        all_losses[task_id] = orig_loss.detach()
        all_losses = self.accelerator.reduce(all_losses, "sum")
        
        if self.step_count > 0 and self.step_count % self.famo_update_frequency == 0:
            prev_task_id = self.prev_inputs.pop("task_id")[0].item()
            with torch.no_grad():
                new_loss = self.eval_loss(model, self.prev_inputs)
                if self.rejected_ids is not None and prev_task_id in self.rejected_ids:
                    new_loss = - self.loss_scale * torch.log(1 + new_loss)
                    # beta = 0.1
                    # new_loss = -F.logsigmoid(beta * new_loss).mean() * 2 / beta 
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
