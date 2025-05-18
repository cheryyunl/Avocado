import os
import tempfile
import shutil
from datasets import Dataset
from huggingface_hub import login, Repository, create_repo, upload_file

def create_dahoas_style_dataset():
    """
    将已转换的数据集以Dahoas格式上传到HuggingFace Hub
    创建一个包含train和test两个parquet文件的单一仓库
    """
    
    print("Loading existing datasets...")
    
    # 加载已存在的数据集
    try:
        # 加载harmless数据集
        train_harmless = Dataset.load_from_disk("converted_harmless")
        test_harmless = Dataset.load_from_disk("converted_harmless_test")
        print(f"Loaded harmless datasets: train={len(train_harmless)}, test={len(test_harmless)}")
        
        # 加载helpful数据集
        train_helpful = Dataset.load_from_disk("converted_helpful")
        test_helpful = Dataset.load_from_disk("converted_helpful_test")
        print(f"Loaded helpful datasets: train={len(train_helpful)}, test={len(test_helpful)}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # 询问用户要上传哪个数据集
    print("\nWhich dataset do you want to upload?")
    print("1. Harmless dataset")
    print("2. Helpful dataset")
    dataset_choice = input("Enter choice (1/2): ").strip()
    
    if dataset_choice == "1":
        train_dataset = train_harmless
        test_dataset = test_harmless
        dataset_type = "harmless"
    elif dataset_choice == "2":
        train_dataset = train_helpful
        test_dataset = test_helpful
        dataset_type = "helpful"
    else:
        print("Invalid choice. Exiting.")
        return
    
    # 询问是否上传到HuggingFace Hub
    print("\nDo you want to upload the dataset to HuggingFace Hub? (y/n)")
    answer = input().strip().lower()
    
    if answer == 'y':
        try:
            # 登录HuggingFace Hub
            print("\nYou need to login to HuggingFace Hub.")
            print("Please enter your HuggingFace token (or press Enter to use the huggingface-cli login):")
            token = input().strip()
            
            if token:
                login(token=token)
                print("Logged in successfully with provided token.")
            else:
                print("Please run 'huggingface-cli login' in a separate terminal and follow the instructions.")
                print("Once logged in, press Enter to continue...")
                input()
            
            # 获取用户名
            print("\nPlease enter your HuggingFace username:")
            username = input().strip()
            
            if not username:
                print("Username cannot be empty. Aborting upload.")
                return
            
            # 询问仓库名称
            print("\nPlease enter a name for the dataset repository (default: hh-rlhf-converted):")
            repo_name = input().strip() or f"hh-rlhf-{dataset_type}-converted"
            
            # 创建完整的仓库名称
            full_repo_name = f"{username}/{repo_name}"
            
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
                
                # 创建存储库
                print(f"\nCreating repository {full_repo_name}...")
                create_repo(full_repo_name, repo_type="dataset", exist_ok=True)
                
                # 克隆存储库
                repo_local_path = os.path.join(tmp_dir, repo_name)
                repo = Repository(repo_local_path, clone_from=full_repo_name)
                
                # 创建data目录
                repo_data_dir = os.path.join(repo_local_path, "data")
                os.makedirs(repo_data_dir, exist_ok=True)
                
                # 复制parquet文件到仓库
                shutil.copy(train_parquet_path, os.path.join(repo_data_dir, "train-00000-of-00001.parquet"))
                shutil.copy(test_parquet_path, os.path.join(repo_data_dir, "test-00000-of-00001.parquet"))
                
                # 创建README.md文件
                readme_path = os.path.join(repo_local_path, "README.md")
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
                    f.write(f"dataset = load_dataset(\"{full_repo_name}\")\n\n")
                    f.write("# Access train split\n")
                    f.write("train_data = dataset[\"train\"]\n\n")
                    f.write("# Access test split\n")
                    f.write("test_data = dataset[\"test\"]\n")
                    f.write("```\n")
                
                # 创建dataset_infos.json文件
                dataset_infos_path = os.path.join(repo_local_path, "dataset_infos.json")
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
                
                # 将更改推送到远程存储库
                print("\nPushing files to repository...")
                repo.git_add(auto_lfs_track=True)
                repo.git_commit("Upload dataset in Dahoas format")
                repo.git_push()
                
                print(f"\nSuccessfully uploaded dataset to {full_repo_name}")
                print(f"You can now use it with: dataset = load_dataset('{full_repo_name}')")
                
        except ImportError:
            print("Required packages are not installed. Please install them with:")
            print("pip install datasets huggingface_hub")
            
        except Exception as e:
            print(f"Error during upload: {e}")
    
    print("\nProcess completed!")

if __name__ == "__main__":
    create_dahoas_style_dataset()