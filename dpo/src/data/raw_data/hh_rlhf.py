# adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py#L82
from dataclasses import dataclass
from typing import Dict, Optional, List

from datasets import load_dataset, concatenate_datasets

from .utils import RawDatasetPreprocessor


def preprocess_anthropic_prompt_and_response(prompt_and_response):
    prompt_and_response = prompt_and_response.replace("\n\nHuman: ", "\n\nHuman:\n")
    prompt_and_response = prompt_and_response.replace("\n\nAssistant: ", "\n\nAssistant:\n")
    return prompt_and_response

def extract_anthropic_prompt_and_response(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:\n"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


@dataclass
class HhRlhfRDP(RawDatasetPreprocessor):
    path: Optional[str] = "Anthropic/hh-rlhf"
    subdatasets: Optional[List[str]] = None

    def __post_init__(self):
        if self.subdatasets is None:
            self.subdatasets = [
                "helpful-base", 
                "helpful-online", 
                "helpful-rejection-sampled",
                "harmless-base"
            ]

    def _get_raw_dataset(self, split):
        all_datasets = []

        for data_dir in self.subdatasets:
            if split == "validation":
                ds = load_dataset(self.path, data_dir=data_dir, split='test')
            else:
                ds = load_dataset(self.path, data_dir=data_dir, split=split)
            all_datasets.append(ds)
            print(f"Successfully loaded {data_dir}, split: {split}, size: {len(ds)}")
        
        if not all_datasets:
            raise ValueError(f"No datasets were successfully loaded for split: {split}")
            
        combined_dataset = concatenate_datasets(all_datasets)
        combined_dataset = combined_dataset.shuffle(seed=42)
        if split == "train":
            return combined_dataset.train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return combined_dataset.train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return combined_dataset
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
        }
    
@dataclass
class MTHhRlhfRDP(RawDatasetPreprocessor):
    path: Optional[str] = "Anthropic/hh-rlhf"
    harmless_subdatasets: Optional[List[str]] = None
    helpful_subdatasets: Optional[List[str]] = None
    
    def __post_init__(self):
        self.harmless_subdatasets = ["harmless-base"]
        
        self.helpful_subdatasets = [
            "helpful-base",
            "helpful-online",
            "helpful-rejection-sampled",
        ]

    def _get_raw_dataset(self, split):
        # 加载所有数据集
        all_datasets = []

        # 加载harmless数据集
        for data_dir in self.harmless_subdatasets:
            if split == "validation":
                ds = load_dataset(self.path, data_dir=data_dir, split='test')
            else:
                ds = load_dataset(self.path, data_dir=data_dir, split=split)
            ds = ds.map(lambda x: {"task_id": 0})  # 添加task_id=0
            all_datasets.append(ds)
            print(f"Successfully loaded {data_dir}, split: {split}, size: {len(ds)}")
        
        # 加载helpful数据集
        for data_dir in self.helpful_subdatasets:
            if split == "validation":
                ds = load_dataset(self.path, data_dir=data_dir, split='test')
            else:
                ds = load_dataset(self.path, data_dir=data_dir, split=split)
            ds = ds.map(lambda x: {"task_id": 1})  # 添加task_id=1
            all_datasets.append(ds)
            print(f"Successfully loaded {data_dir}, split: {split}, size: {len(ds)}")
        
        if not all_datasets:
            raise ValueError(f"No datasets were successfully loaded for split: {split}")
            
        # 合并所有数据集
        combined_dataset = concatenate_datasets(all_datasets)
        
        # 计算任务数量
        task_id_counts = {}
        for example in combined_dataset:
            task_id = example["task_id"]
            if task_id not in task_id_counts:
                task_id_counts[task_id] = 0
            task_id_counts[task_id] += 1
        
        print(f"原始任务分布: {task_id_counts}")
        
        # 分离不同任务的数据
        task0_data = combined_dataset.filter(lambda x: x["task_id"] == 0)
        task1_data = combined_dataset.filter(lambda x: x["task_id"] == 1)
        
        # 为了平衡，计算目标大小
        task0_size = len(task0_data)
        task1_size = len(task1_data)
        
        # 确定目标大小（小数据集的2倍，但不超过大数据集的大小）
        if task0_size <= task1_size:
            target_size = min(task0_size * 2, task1_size)
        else:
            target_size = min(task1_size * 2, task0_size)
            
        print(f"目标均衡大小: {target_size}")
        
        # 重采样task0数据
        if task0_size < target_size:
            # 上采样
            repeat_factor = target_size / task0_size
            repeat_times = int(repeat_factor)
            remainder = target_size - (repeat_times * task0_size)
            
            if repeat_times > 0:
                repeated_datasets = [task0_data] * repeat_times
                if remainder > 0:
                    remainder_dataset = task0_data.select(range(remainder))
                    repeated_datasets.append(remainder_dataset)
                task0_data = concatenate_datasets(repeated_datasets)
            elif remainder > 0:
                task0_data = task0_data.select(range(target_size))
                
            task0_data = task0_data.shuffle(seed=42)
            print(f"重采样后task0大小: {len(task0_data)}")
        elif task0_size > target_size:
            # 下采样
            task0_data = task0_data.select(range(target_size))
            task0_data = task0_data.shuffle(seed=42)
            print(f"下采样后task0大小: {len(task0_data)}")
            
        # 重采样task1数据
        if task1_size < target_size:
            # 上采样
            repeat_factor = target_size / task1_size
            repeat_times = int(repeat_factor)
            remainder = target_size - (repeat_times * task1_size)
            
            if repeat_times > 0:
                repeated_datasets = [task1_data] * repeat_times
                if remainder > 0:
                    remainder_dataset = task1_data.select(range(remainder))
                    repeated_datasets.append(remainder_dataset)
                task1_data = concatenate_datasets(repeated_datasets)
            elif remainder > 0:
                task1_data = task1_data.select(range(target_size))
                
            task1_data = task1_data.shuffle(seed=42)
            print(f"重采样后task1大小: {len(task1_data)}")
        elif task1_size > target_size:
            # 下采样
            task1_data = task1_data.select(range(target_size))
            task1_data = task1_data.shuffle(seed=42)
            print(f"下采样后task1大小: {len(task1_data)}")
        
        # 按顺序排列：前半段task_id=0，后半段task_id=1
        balanced_dataset = concatenate_datasets([task0_data, task1_data])
        print(f"最终均衡数据集大小: {len(balanced_dataset)}")
        
        # 处理split
        if split == "train":
            return balanced_dataset.train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return balanced_dataset.train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return balanced_dataset
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
            "task_id":  example["task_id"]
        }

class TestHhRlhfRDP(RawDatasetPreprocessor):
    path: Optional[str] = "Anthropic/hh-rlhf"
    subdatasets: Optional[List[str]] = None

    def __post_init__(self):
        self.subdatasets = [
            "helpful-base", 
            "helpful-online", 
            "helpful-rejection-sampled",
            "harmless-base"
        ]

    def _get_raw_dataset(self, split):
        all_datasets = []

        for data_dir in self.subdatasets:
            if split == "validation":
                ds = load_dataset(self.path, data_dir=data_dir, split='test')
            else:
                ds = load_dataset(self.path, data_dir=data_dir, split=split)
            all_datasets.append(ds)
            print(f"Successfully loaded {data_dir}, split: {split}, size: {len(ds)}")
        
        if not all_datasets:
            raise ValueError(f"No datasets were successfully loaded for split: {split}")
            
        combined_dataset = concatenate_datasets(all_datasets)
        total_size = len(combined_dataset)
        mid_point = total_size // 2
        
        # 为前半部分设置task_id=0，后半部分设置task_id=1
        first_half = combined_dataset.select(range(mid_point))
        first_half = first_half.map(lambda x: {"task_id": 0})
        
        second_half = combined_dataset.select(range(mid_point, total_size))
        second_half = second_half.map(lambda x: {"task_id": 1})
        
        # 重新合并数据集
        combined_dataset = concatenate_datasets([first_half, second_half])
        combined_dataset = combined_dataset.shuffle(seed=42)
        if split == "train":
            return combined_dataset.train_test_split(test_size=0.1, seed=0)["train"]
        elif split == "validation":
            return combined_dataset.train_test_split(test_size=0.1, seed=0)["test"]
        elif split == "test":
            return combined_dataset
        else:
            raise NotImplementedError

    def _dataset_to_preference_formatter(self, example) -> Dict[str, str]:
        example["chosen"]   = preprocess_anthropic_prompt_and_response(example["chosen"])
        example["rejected"] = preprocess_anthropic_prompt_and_response(example["rejected"])
        prompt = extract_anthropic_prompt_and_response(example["chosen"])
        return {
            "prompt":   prompt,
            "chosen":   example["chosen"][len(prompt) :],
            "rejected": example["rejected"][len(prompt) :],
            "task_id":  example["task_id"]
        }


if __name__ == "__main__":
    train_dataset      = HhRlhfRDP(num_proc=1).get_preference_dataset(split="train")
    validation_dataset = HhRlhfRDP(num_proc=1).get_preference_dataset(split="validation")
    test_dataset       = HhRlhfRDP(num_proc=1).get_preference_dataset(split="test")
