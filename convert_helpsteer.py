from datasets import load_dataset, Dataset
import os
import json
import argparse

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

def dataset_to_preference_formatter(example, dimension, prompt_template="Human: {raw_prompt} Assistant:"):
    chosen_id = example[f"{dimension}_chosen_id"]
    return {
        "prompt": prompt_template.format(raw_prompt=example["prompt"]),
        "chosen": example[f"response_{chosen_id}"],
        "rejected": example[f"response_{1-chosen_id}"],
        "response": example[f"response_{chosen_id}"],  # 添加response字段
    }

def process_helpsteer_dataset(split, dimension, num_proc=8):
    """处理HelpSteer数据集为指定维度的DPO偏好对"""
    # 加载原始数据集
    print(f"Loading HelpSteer {split} dataset...")
    dataset = load_dataset("nvidia/HelpSteer", split=split)
    
    # 转换为偏好对
    print(f"Transforming dataset to preference pairs...")
    dataset = dataset.map(
        helpsteer_transform_to_preference,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    
    # 过滤没有偏好的样本
    print(f"Filtering preferences for dimension: {dimension}...")
    dataset = dataset.filter(lambda x: x[f"{dimension}_chosen_id"] != -1)
    
    # 格式化为标准DPO格式
    print(f"Formatting dataset to standard DPO format...")
    dataset = dataset.map(
        lambda x: dataset_to_preference_formatter(x, dimension),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    
    print(f"Created {dimension} {split} dataset with {len(dataset)} examples")
    return dataset

def save_dataset_as_parquet(dataset, output_dir, file_name):
    """将数据集保存为parquet文件"""
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet文件
    file_path = os.path.join(output_dir, file_name)
    dataset.to_parquet(file_path)
    print(f"Saved dataset to {file_path}")
    return file_path

def main():
    parser = argparse.ArgumentParser(description="Convert HelpSteer dataset to DPO format")
    parser.add_argument("--dimensions", type=str, nargs="+", default=["helpfulness", "correctness", "coherence"],
                        help="Dimensions to process (default: helpfulness correctness coherence)")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of processes for parallel processing")
    parser.add_argument("--output_base", type=str, default="helpsteer",
                        help="Base name for output directories")
    
    args = parser.parse_args()
    
    for dimension in args.dimensions:
        # 创建数据目录
        dataset_dir = f"{args.output_base}-{dimension}"
        data_dir = os.path.join(dataset_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 处理训练集
        train_dataset = process_helpsteer_dataset("train", dimension, args.num_proc)
        save_dataset_as_parquet(train_dataset, data_dir, "train-00000-of-00001.parquet")
        
        # 处理验证集并存为test
        val_dataset = process_helpsteer_dataset("validation", dimension, args.num_proc)
        save_dataset_as_parquet(val_dataset, data_dir, "test-00000-of-00001.parquet")
        
        # 创建dataset_infos.json文件
        dataset_infos = {
            "default": {
                "description": f"HelpSteer dataset processed for {dimension} preference learning",
                "citation": "",
                "homepage": "https://huggingface.co/datasets/nvidia/HelpSteer",
                "license": "",
                "features": {
                    "prompt": {"dtype": "string", "_type": "Value"},
                    "response": {"dtype": "string", "_type": "Value"},
                    "chosen": {"dtype": "string", "_type": "Value"},
                    "rejected": {"dtype": "string", "_type": "Value"}
                },
                "splits": {
                    "train": {
                        "name": "train",
                        "num_bytes": 0,
                        "num_examples": len(train_dataset),
                        "dataset_name": f"{args.output_base}-{dimension}"
                    },
                    "test": {
                        "name": "test",
                        "num_bytes": 0,
                        "num_examples": len(val_dataset),
                        "dataset_name": f"{args.output_base}-{dimension}"
                    }
                }
            }
        }
        
        # 保存dataset_infos.json
        with open(os.path.join(dataset_dir, "dataset_infos.json"), "w") as f:
            json.dump(dataset_infos, f, indent=2)
            
        # 创建README.md
        readme = f"""# {args.output_base.capitalize()}-{dimension}

        This dataset is derived from NVIDIA's HelpSteer dataset, processed specifically for preference learning on the {dimension} dimension.

        - Train split: {len(train_dataset)} examples
        - Test split: {len(val_dataset)} examples

        ## Format

        Each example contains the following fields:
        - `prompt`: Question with "Human:" prefix and "Assistant:" suffix
        - `chosen`: The response with higher {dimension} score
        - `rejected`: The response with lower {dimension} score
        - `response`: Same as chosen

        ## Usage

        ```python
        from datasets import load_dataset

        # Load the dataset
        dataset = load_dataset("your-username/{args.output_base}-{dimension}")

        # Access train split
        train_data = dataset["train"]

        # Access test split
        test_data = dataset["test"]
        """
        with open(os.path.join(dataset_dir, "README.md"), "w") as f:
            f.write(readme)
            
        # 打印示例以检查格式
        print("\n检查样本格式示例:")
        sample = train_dataset[0]
        print(f"Prompt: {sample['prompt']}")
        print(f"Response: {sample['response']}")
        print(f"Chosen: {sample['chosen']}")
        print(f"Rejected: {sample['rejected']}")

        print("\n上传指南:")
        print("要上传数据集到HuggingFace Hub，请运行:")
        print("python upload_to_huggingface.py --repos " + " ".join([f"{args.output_base}-{dim}" for dim in args.dimensions]))

        print("\nConversion completed successfully!")

if __name__ == "__main__":
    main()
