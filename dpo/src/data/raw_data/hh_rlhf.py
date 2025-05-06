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


if __name__ == "__main__":
    train_dataset      = HhRlhfRDP(num_proc=1).get_preference_dataset(split="train")
    validation_dataset = HhRlhfRDP(num_proc=1).get_preference_dataset(split="validation")
    test_dataset       = HhRlhfRDP(num_proc=1).get_preference_dataset(split="test")
