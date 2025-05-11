from typing import Dict, Optional, List, Literal
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from typing import Literal, Optional

from .utils import RawDatasetPreprocessor
from src.utils import print_local_main


def helpsteer_transform_to_preference(batched_sample):
    def chosen_id(score_0, score_1):
        if score_0 < score_1:
            return 1
        elif score_0 > score_1:
            return 0
        else:
            return -1

    finegrained_dimensions = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
    dimensions = finegrained_dimensions + ("overall",)

    debatched_sample = [{k:batched_sample[k][i] for k in batched_sample.keys()} for i in range(len(batched_sample["prompt"]))]

    new_batched_sample = {
        "prompt": [],
        "response_0": [],
        "response_1": [],
        **{f"{dimension}_chosen_id": [] for dimension in dimensions}
    }
    mini_debatch = []
    for i, sample in enumerate(debatched_sample):
        mini_debatch.append(sample)
        if i != len(debatched_sample) - 1 and sample["prompt"] == debatched_sample[i+1]["prompt"]:
            continue

        for j in range(len(mini_debatch)):
            for k in range(j+1, len(mini_debatch)):
                new_batched_sample["prompt"].append(mini_debatch[j]["prompt"])
                new_batched_sample["response_0"].append(mini_debatch[j]["response"])
                new_batched_sample["response_1"].append(mini_debatch[k]["response"])
                new_batched_sample["overall_chosen_id"].append(
                    chosen_id(
                        sum(mini_debatch[j][dimension] for dimension in finegrained_dimensions),
                        sum(mini_debatch[k][dimension] for dimension in finegrained_dimensions),
                    )
                )
                for dimension in finegrained_dimensions:
                    new_batched_sample[f"{dimension}_chosen_id"].append(
                        chosen_id(
                            mini_debatch[j][dimension], 
                            mini_debatch[k][dimension],
                        )
                    )

        mini_debatch = []

    return new_batched_sample


@dataclass
class HelpSteerRDP(RawDatasetPreprocessor):
    path: Optional[str] = "nvidia/HelpSteer"
    # None for sft
    dimension: Optional[Literal["overall", "helpfulness", "correctness", "coherence", "complexity", "verbosity"]] = None

    def _get_raw_dataset(self, split):
        if split == "train":
            return load_dataset(self.path, split="train")
        elif split == "validation":
            return load_dataset(self.path, split="validation")
        elif split == "test":
            raise NotImplementedError("test split not implemented for helpsteer")
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        chosen_id = example[f"{self.dimension}_chosen_id"]
        return {
            "raw_prompt": example["prompt"],
            "prompt":   self.prompt_template.format(raw_prompt=example["prompt"]),
            "chosen":   example[f"response_{chosen_id}"],
            "rejected": example[f"response_{1-chosen_id}"],
        }

    def get_preference_dataset(self, split):
        assert self.dimension, "preference dimension has to be specified"
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to preference...")
        dataset = dataset.map(
            helpsteer_transform_to_preference,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )
        print_local_main("filtering preference...")
        dataset = dataset.filter(lambda x: x[f"{self.dimension}_chosen_id"] != -1)
        print_local_main("mapping dataset to standard format...")
        return dataset.map(self._dataset_to_preference_formatter, num_proc=self.num_proc, remove_columns=dataset.column_names)

    def get_sft_dataset(self, split, **kwargs):
        if self.dimension:
            return super().get_sft_dataset(split, **kwargs)
        dataset = self._get_raw_dataset(split)
        if self.sanity_check: 
            dataset = dataset.select(range(min(len(dataset), 100)))
        print_local_main("mapping raw dataset to sft...")
        return dataset.map(
            lambda sample: {
                "raw_prompt": sample["prompt"],
                "prompt": self.prompt_template.format(raw_prompt=sample["prompt"]), 
                "response": sample["response"], 
            },
            num_proc=self.num_proc,
            remove_columns=dataset.column_names,
        )

@dataclass
class MTHelpSteerRDP(RawDatasetPreprocessor):
    path: Optional[str] = "nvidia/HelpSteer"
    dimensions: Optional[List[Literal["helpfulness", "correctness", "coherence", "complexity"]]] = None
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = ["helpfulness", "correctness", "coherence", "complexity"]
    
    def _get_raw_dataset(self, split):
        dimension_datasets = []
        
        for i, dimension in enumerate(self.dimensions):
            dimension_rdp = HelpSteerRDP(
                path=self.path,
                dimension=dimension,
                prompt_template=self.prompt_template,
                num_proc=self.num_proc,
                sanity_check=self.sanity_check
            )
            
            ds = dimension_rdp.get_preference_dataset(split=split)
            print_local_main(f"Successfully loaded dimension: {dimension}, split: {split}, size: {len(ds)}")
            
            ds = ds.map(lambda x: {"task_id": i})
            dimension_datasets.append(ds)
        
        if not dimension_datasets:
            raise ValueError(f"No datasets were successfully loaded for split: {split}")
        
        dimension_sizes = [len(ds) for ds in dimension_datasets]
        print_local_main(f"Original dimension data sizes: {dict(zip(self.dimensions, dimension_sizes))}")
        
        target_size = max(dimension_sizes)
        print_local_main(f"Target balanced size: {target_size}")
        
        balanced_datasets = []
        for i, (dimension, ds) in enumerate(zip(self.dimensions, dimension_datasets)):
            dimension_size = len(ds)
            
            if dimension_size < target_size:
                repeat_factor = target_size / dimension_size
                repeat_times = int(repeat_factor)
                remainder = target_size - (repeat_times * dimension_size)
                
                if repeat_times > 0:
                    repeated_datasets = [ds] * repeat_times
                    if remainder > 0:
                        remainder_dataset = ds.select(range(remainder))
                        repeated_datasets.append(remainder_dataset)
                    balanced_ds = concatenate_datasets(repeated_datasets)
                elif remainder > 0:
                    balanced_ds = ds.select(range(target_size))
                
                balanced_ds = balanced_ds.shuffle(seed=42)
                print_local_main(f"Resampled {dimension} data size: {len(balanced_ds)}")
            
            elif dimension_size > target_size:
                balanced_ds = ds.select(range(target_size))
                balanced_ds = balanced_ds.shuffle(seed=42)
                print_local_main(f"Downsized {dimension} data size: {len(balanced_ds)}")
            
            else:
                balanced_ds = ds
            
            balanced_datasets.append(balanced_ds)
        
        combined_dataset = concatenate_datasets(balanced_datasets)
        print_local_main(f"Final combined data size: {len(combined_dataset)}")
        
        return combined_dataset
    
    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
            "task_id": example["task_id"]
        }

@dataclass
class MixedHelpSteerRDP(RawDatasetPreprocessor):
    path: Optional[str] = "nvidia/HelpSteer"
    dimensions: Optional[List[Literal["helpfulness", "correctness", "coherence", "complexity"]]] = None
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = ["helpfulness", "correctness", "coherence", "complexity"]
    
    def _get_raw_dataset(self, split):
        dimension_datasets = []
        
        for i, dimension in enumerate(self.dimensions):
            dimension_rdp = HelpSteerRDP(
                path=self.path,
                dimension=dimension,
                prompt_template=self.prompt_template,
                num_proc=self.num_proc,
                sanity_check=self.sanity_check
            )
            
            ds = dimension_rdp.get_preference_dataset(split=split)
            print_local_main(f"Successfully loaded dimension: {dimension}, split: {split}, size: {len(ds)}")
            dimension_datasets.append(ds)
        
        if not dimension_datasets:
            raise ValueError(f"No datasets were successfully loaded for split: {split}")
        
        dimension_sizes = [len(ds) for ds in dimension_datasets]
        print_local_main(f"Original dimension data sizes: {dict(zip(self.dimensions, dimension_sizes))}")
        
        target_size = max(dimension_sizes)
        print_local_main(f"Target balanced size: {target_size}")
        
        balanced_datasets = []
        for i, (dimension, ds) in enumerate(zip(self.dimensions, dimension_datasets)):
            dimension_size = len(ds)
            
            if dimension_size < target_size:
                repeat_factor = target_size / dimension_size
                repeat_times = int(repeat_factor)
                remainder = target_size - (repeat_times * dimension_size)
                
                if repeat_times > 0:
                    repeated_datasets = [ds] * repeat_times
                    if remainder > 0:
                        remainder_dataset = ds.select(range(remainder))
                        repeated_datasets.append(remainder_dataset)
                    balanced_ds = concatenate_datasets(repeated_datasets)
                elif remainder > 0:
                    balanced_ds = ds.select(range(target_size))
                
                print_local_main(f"Resampled {dimension} data size: {len(balanced_ds)}")
            
            elif dimension_size > target_size:
                balanced_ds = ds.select(range(target_size))
                print_local_main(f"Downsized {dimension} data size: {len(balanced_ds)}")
            
            else:
                balanced_ds = ds
            
            balanced_datasets.append(balanced_ds)
        
        combined_dataset = concatenate_datasets(balanced_datasets)
        combined_dataset = combined_dataset.shuffle(seed=42)
        print_local_main(f"Final combined data size: {len(combined_dataset)}")
        
        return combined_dataset
    
    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }
    
if __name__ == "__main__":
    num_proc = 4
    helpful_dataset = HelpSteerRDP(dimension="helpfulness", num_proc=num_proc).get_preference_dataset(split="train")
    overall_dataset = HelpSteerRDP(dimension="overall", num_proc=num_proc).get_preference_dataset(split="train")
    sft_dataset     = HelpSteerRDP(num_proc=num_proc).get_sft_dataset(split="train")
    breakpoint()
