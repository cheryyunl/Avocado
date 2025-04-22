import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
from mt_dataset import build_mt_dataset, MultiDatasetSampler
from utils import load_main_tokenizer, Instructions_summary
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, set_seed, DataCollatorForCompletionOnlyLM

# Import your MultiDatasetSampler and build_mt_dataset function here

def test_multi_dataset_sampler():
    # Load the datasets
    tokenizer = load_main_tokenizer("meta-llama/Llama-2-7b-hf")
    dataset, sizes = build_mt_dataset(tokenizer, split='train', size=1000, seed=42)

    batch_size = 8  # Must be divisible by number of datasets (2) and number of simulated GPUs (4)
    num_simulated_gpus = 4

    # Create samplers for each simulated GPU
    samplers = [
        MultiDatasetSampler(
            dataset_sizes=sizes, 
            batch_size=batch_size, 
            shuffle=True, 
            seed=42
        ) for _ in range(num_simulated_gpus)
    ]

    
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)

    # Simulate sampling for each GPU
    all_samples = defaultdict(list)
    for gpu_id, sampler in enumerate(samplers):
        sampler.num_replicas = num_simulated_gpus
        sampler.rank = gpu_id
        
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=4)
        
        for batch in dataloader:
            all_samples[gpu_id].extend(batch['input_ids'].tolist())

    # Verify the results
    total_samples = sum(len(samples) for samples in all_samples.values())
    expected_samples_per_gpu = len(dataset) // num_simulated_gpus

    for gpu_id, samples in all_samples.items():
        # Check number of samples
        assert len(samples) == expected_samples_per_gpu, \
            f"GPU {gpu_id} has {len(samples)} samples, expected {expected_samples_per_gpu}"

        # Check distribution of samples from each dataset
        dataset1_samples = sum(1 for sample in samples if sample[0] < sizes[0])
        dataset2_samples = sum(1 for sample in samples if sample[0] >= sizes[0])

        expected_per_dataset = expected_samples_per_gpu // 2
        tolerance = expected_per_dataset * 0.1  # Allow 10% tolerance

        assert abs(dataset1_samples - expected_per_dataset) <= tolerance, \
            f"GPU {gpu_id} has {dataset1_samples} samples from dataset 1, expected around {expected_per_dataset}"
        assert abs(dataset2_samples - expected_per_dataset) <= tolerance, \
            f"GPU {gpu_id} has {dataset2_samples} samples from dataset 2, expected around {expected_per_dataset}"

        # Check that each batch has the correct mix of samples
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            dataset1_in_batch = sum(1 for sample in batch if sample[0] < sizes[0])
            dataset2_in_batch = sum(1 for sample in batch if sample[0] >= sizes[0])
            
            assert dataset1_in_batch == batch_size // 2, \
                f"Batch on GPU {gpu_id} has {dataset1_in_batch} samples from dataset 1, expected {batch_size // 2}"
            assert dataset2_in_batch == batch_size // 2, \
                f"Batch on GPU {gpu_id} has {dataset2_in_batch} samples from dataset 2, expected {batch_size // 2}"

    print("All tests passed successfully!")

if __name__ == "__main__":
    test_multi_dataset_sampler()