import os
import tempfile
from datasets import Dataset
from huggingface_hub import login, HfApi

def upload_to_existing_dataset():
    """
    将已转换的数据集以Dahoas格式上传到已有的HuggingFace数据集仓库
    """
    
    print("Loading existing datasets...")
    
    # 加载已存在的数据集
    try:
        # 加载harmless或helpful数据集
        dataset_type = input("Which dataset type to upload? (harmless/helpful): ").strip().lower()
        
        if dataset_type == "harmless":
            train_dataset = Dataset.load_from_disk("converted_harmless")
            test_dataset = Dataset.load_from_disk("converted_harmless_test")
        elif dataset_type == "helpful":
            train_dataset = Dataset.load_from_disk("converted_helpful")
            test_dataset = Dataset.load_from_disk("converted_helpful_test")
        else:
            print("Invalid choice. Exiting.")
            return
            
        print(f"Loaded {dataset_type} datasets: train={len(train_dataset)}, test={len(test_dataset)}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # 询问仓库名称
    print("\nPlease enter your HuggingFace username:")
    username = input().strip()
    
    if not username:
        print("Username cannot be empty. Aborting upload.")
        return
    
    # 询问仓库名称
    print("\nPlease enter the name of your existing dataset repository:")
    repo_name = input().strip()
    
    if not repo_name:
        print("Repository name cannot be empty. Aborting upload.")
        return
    
    # 完整的仓库名称和ID
    full_repo_name = f"{username}/{repo_name}"
    repo_id = f"{username}/{repo_name}"  # 注意：没有datasets/前缀
    
    # 询问是否继续上传
    print(f"\nWill upload to dataset repository: {full_repo_name}")
    print("Continue? (y/n)")
    answer = input().strip().lower()
    
    if answer != 'y':
        print("Upload cancelled.")
        return
    
    # 登录HuggingFace Hub
    print("\nLogging in to HuggingFace Hub...")
    login()
    
    try:
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 创建data目录
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # 将数据集转换为parquet格式并保存
            print("\nConverting datasets to parquet format...")
            
            train_parquet_path = os.path.join(data_dir, "train-00000-of-00001.parquet")
            test_parquet_path = os.path.join(data_dir, "test-00000-of-00001.parquet")
            
            train_dataset.to_parquet(train_parquet_path)
            test_dataset.to_parquet(test_parquet_path)
            
            print(f"Saved train dataset ({len(train_dataset)} examples) to {train_parquet_path}")
            print(f"Saved test dataset ({len(test_dataset)} examples) to {test_parquet_path}")
            
            # 创建README.md文件
            readme_path = os.path.join(tmp_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"# {repo_name}\n\n")
                f.write(f"This dataset is a converted version of Anthropic's HH-RLHF dataset in Dahoas/full-hh-rlhf format.\n\n")
                f.write(f"- Train split: {len(train_dataset)} examples\n")
                f.write(f"- Test split: {len(test_dataset)} examples\n\n")
                f.write("## Format\n\n")
                f.write("Each example contains the following fields:\n")
                f.write("- `prompt`: Complete conversation history including the last 'Assistant:' prefix\n")
                f.write("- `chosen`: The preferred assistant response (without 'Assistant:' prefix)\n")
                f.write("- `rejected`: The dispreferred assistant response (without 'Assistant:' prefix)\n")
                f.write("- `response`: Same as chosen\n\n")
                f.write("## Usage\n\n")
                f.write("```python\n")
                f.write("from datasets import load_dataset\n\n")
                f.write(f"# Load the dataset\n")
                f.write(f"dataset = load_dataset(\"{username}/{repo_name}\")\n\n")
                f.write("# Access train split\n")
                f.write("train_data = dataset[\"train\"]\n\n")
                f.write("# Access test split\n")
                f.write("test_data = dataset[\"test\"]\n")
                f.write("```\n")
            
            # 创建dataset_infos.json文件
            dataset_infos_path = os.path.join(tmp_dir, "dataset_infos.json")
            with open(dataset_infos_path, "w") as f:
                f.write('{\n')
                f.write('  "default": {\n')
                f.write('    "description": "Converted HH-RLHF dataset in Dahoas format",\n')
                f.write(f'    "citation": "",\n')
                f.write('    "homepage": "https://huggingface.co/datasets/Anthropic/hh-rlhf",\n')
                f.write('    "license": "",\n')
                f.write('    "features": {\n')
                f.write('      "prompt": {"dtype": "string", "id": null, "_type": "Value"},\n')
                f.write('      "response": {"dtype": "string", "id": null, "_type": "Value"},\n')
                f.write('      "chosen": {"dtype": "string", "id": null, "_type": "Value"},\n')
                f.write('      "rejected": {"dtype": "string", "id": null, "_type": "Value"}\n')
                f.write('    },\n')
                f.write('    "supervised_keys": null,\n')
                f.write('    "task_templates": null,\n')
                f.write(f'    "builder_name": "{repo_name}",\n')
                f.write('    "config_name": "default",\n')
                f.write('    "version": {\n')
                f.write('      "version_str": "1.0.0",\n')
                f.write('      "description": null,\n')
                f.write('      "major": 1,\n')
                f.write('      "minor": 0,\n')
                f.write('      "patch": 0\n')
                f.write('    },\n')
                f.write(f'    "splits": {{\n')
                f.write(f'      "train": {{\n')
                f.write(f'        "name": "train",\n')
                f.write(f'        "num_bytes": {train_dataset.info.dataset_size if hasattr(train_dataset.info, "dataset_size") else 0},\n')
                f.write(f'        "num_examples": {len(train_dataset)},\n')
                f.write(f'        "dataset_name": "{repo_name}"\n')
                f.write(f'      }},\n')
                f.write(f'      "test": {{\n')
                f.write(f'        "name": "test",\n')
                f.write(f'        "num_bytes": {test_dataset.info.dataset_size if hasattr(test_dataset.info, "dataset_size") else 0},\n')
                f.write(f'        "num_examples": {len(test_dataset)},\n')
                f.write(f'        "dataset_name": "{repo_name}"\n')
                f.write(f'      }}\n')
                f.write(f'    }}\n')
                f.write('  }\n')
                f.write('}\n')
            
            # 直接使用HfApi.upload_folder上传
            print("\nUploading files to repository...")
            api = HfApi()
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type="dataset"
            )
            
            print(f"\nSuccessfully uploaded dataset to {full_repo_name}")
            print(f"You can now use it with: dataset = load_dataset(\"{username}/{repo_name}\")")
            
    except Exception as e:
        print(f"Error during upload: {e}")
    
    print("\nProcess completed!")

if __name__ == "__main__":
    upload_to_existing_dataset()