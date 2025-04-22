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


class LinearSFTTrainer(SFTTrainer):
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
        self._signature_columns = ["input_ids", "labels", "attention_mask", "task_id"]
        self.compute_loss_func = None
        self.model_accepts_loss_kwargs = False
        self.args = args
        self.args.average_tokens_across_devices = True
        self.n_tasks = n_tasks
        self.rejected_ids = rejected_ids

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
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:

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
                orig_loss = - torch.log(1 + orig_loss)
            loss = orig_loss 
        
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
        
        if self.accelerator.is_main_process:
                wandb.log({
                    **{f'task_{task_id}_loss': all_losses[task_id].item() for task_id in range(self.n_tasks)},
                })

        del inputs

        return loss.detach()
