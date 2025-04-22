from datasets import load_dataset, concatenate_datasets, load_from_disk
import os
from itertools import accumulate
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, DistributedSampler
import random
from transformers import AutoTokenizer
import math
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, set_seed, DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling, PreTrainedTokenizerBase
import dataclasses
import importlib.resources as pkg_resources
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pdb


def build_mt_dataset(path, tokenizer, split='train', size=None, seed=42):
    ds_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    ds_helpful_base = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    ds_helpful_online = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-online", split=split)
    ds_helpful_rs = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-rejection-sampled", split=split)

    ds_helpful = concatenate_datasets([ds_helpful_base, ds_helpful_online, ds_helpful_rs])

    # Add task_id to each dataset
    ds_harmless = ds_harmless.map(lambda x: {"task_id": 0})
    ds_helpful = ds_helpful.map(lambda x: {"task_id": 1})

    ds_harmless = ds_harmless.shuffle(seed=seed)
    ds_helpful = ds_helpful .shuffle(seed=seed)
    ds = concatenate_datasets([ds_harmless, ds_helpful])
    
    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample

    ds_chosen = ds.map(tokenize, batched=False, num_proc=30)
    ds_concat = ds_chosen
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected'])
    ds_concat.set_format(type="torch")

    return ds_concat
    
def build_mt_reject_dataset(path, tokenizer, split='train', size=None, seed=42):
    ds_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    ds_helpful_base = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    ds_helpful_online = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-online", split=split)
    ds_helpful_rs = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-rejection-sampled", split=split)

    ds_helpful = concatenate_datasets([ds_helpful_base, ds_helpful_online, ds_helpful_rs])

    # Add task_id to each dataset
    ds_harmless = ds_harmless.map(lambda x: {"task_id": 0})
    ds_helpful = ds_helpful.map(lambda x: {"task_id": 1})

    ds_harmless = ds_harmless.shuffle(seed=seed)
    ds_helpful = ds_helpful .shuffle(seed=seed)
    ds = concatenate_datasets([ds_harmless, ds_helpful])
    
    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample

    def reject_tokenize(sample):
        sample['text'] = sample['rejected']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample

    ds_harmful_processed = ds_harmless.map(reject_tokenize, batched=False, num_proc=30)
    ds_helpful_processed = ds_helpful.map(tokenize, batched=False, num_proc=30)
    ds_concat = concatenate_datasets([ds_harmful_processed, ds_helpful_processed])
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected'])
    ds_concat.set_format(type="torch")

    return ds_concat
    
def build_mt_all_dataset(path, tokenizer, split='train', size=None, seed=42):
    ds_harmless = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
    ds_helpful_base = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
    ds_helpful_online = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-online", split=split)
    ds_helpful_rs = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-rejection-sampled", split=split)

    ds_helpful = concatenate_datasets([ds_helpful_base, ds_helpful_online, ds_helpful_rs])

    ds_harmless = ds_harmless.shuffle(seed=seed)
    ds_helpful = ds_helpful.shuffle(seed=seed)
    
    def tokenize(sample):
        sample['text'] = sample['chosen']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample

    def reject_tokenize(sample):
        sample['text'] = sample['rejected']
        split_text = sample['text'].split('\n\nAssistant:')
        sample['prompt'] = '\n\nAssistant:'.join(split_text[:-1]) + ' ' + '\n\nAssistant:'
        sample['response'] = split_text[-1].strip()
        sample["input_ids"] = tokenizer.encode(sample["text"]) + [tokenizer.eos_token_id]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["task_id"] = torch.tensor([sample["task_id"]], dtype=torch.long)
        return sample

    ds_harmless_processed = ds_harmless.map(tokenize, batched=False, num_proc=30)
    ds_harmful_processed = ds_harmless.map(reject_tokenize, batched=False, num_proc=30)  
    ds_helpful_processed = ds_helpful.map(tokenize, batched=False, num_proc=30)
    
    # set task_id
    ds_harmless_processed = ds_harmless_processed.map(lambda x: {"task_id": 0})
    ds_harmful_processed = ds_harmful_processed.map(lambda x: {"task_id": 1})
    ds_helpful_processed = ds_helpful_processed.map(lambda x: {"task_id": 2})
    
    if size is not None:
        size_per_category = size // 3
        ds_harmless_processed = ds_harmless_processed.select(range(min(size_per_category, len(ds_harmless_processed))))
        ds_harmful_processed = ds_harmful_processed.select(range(min(size_per_category, len(ds_harmful_processed))))
        ds_helpful_processed = ds_helpful_processed.select(range(min(size_per_category, len(ds_helpful_processed))))
    
    ds_concat = concatenate_datasets([ds_harmless_processed, ds_harmful_processed, ds_helpful_processed])
    ds_concat = ds_concat.filter(lambda x: len(x["input_ids"]) <= 512 and len(x["input_ids"]) >= 8)
    ds_concat = ds_concat.remove_columns(['chosen', 'rejected'])
    ds_concat.set_format(type="torch")

    return ds_concat

class MTDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        task_ids = None
        if all('task_id' in example for example in examples):
            task_ids = [example.pop('task_id') for example in examples]  # Remove task_id from examples

        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                
                if response_token_ids_start_idx is None:
                    for idx in np.where(batch["labels"][i] == self.response_token_ids[1])[0]:
                        if (
                            self.response_token_ids[1:]
                             == batch["labels"][i][idx : idx + len(self.response_token_ids[1:])].tolist()
                        ):
                            response_token_ids_start_idx = idx

                    if response_token_ids_start_idx is None:
                        warnings.warn(
                            f"Could not find response key `{self.response_template}` in the "
                            f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                            f"This instance will be ignored in loss calculation. "
                            f"Note, if this happens often, consider increasing the `max_seq_length`."
                        )
                        batch["labels"][i, :] = self.ignore_index
                    else:
                        response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids[1:])
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                # Make pytorch loss function ignore all tokens up through the end of the response key
                batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index
        
        batch['task_id'] = torch.tensor(task_ids, dtype=torch.long)

        return batch
    
def split_dataset(dataset):
    task_ids = dataset['task_id'].squeeze()
    if isinstance(task_ids, torch.Tensor):
        task_ids = task_ids.cpu()  
    
    unique_tasks = torch.unique(task_ids)
    max_task = unique_tasks.max().item()
    
    task_counts = torch.zeros(max_task + 1, dtype=torch.long)
    for i in range(max_task + 1):
        task_counts[i] = (task_ids == i).sum()
    
    start = 0
    datasets = []
    for size in task_counts:
        end = start + size
        datasets.append(dataset.select(range(start, end)))
        start = end
    
    assert sum(len(d) for d in datasets) == len(dataset)
    return datasets
