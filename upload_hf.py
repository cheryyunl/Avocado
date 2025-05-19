
import argparse
from huggingface_hub import create_repo, HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload datasets to HuggingFace Hub")
    parser.add_argument("--repos", type=str, nargs="+", required=True, 
                        help="Repository names to upload (local directory names)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token, if not provided will use CLI login")
    parser.add_argument("--username", type=str, default=None,
                        help="HuggingFace username, will prompt if not provided")
    
    args = parser.parse_args()
    
    # 登录HuggingFace
    if args.token:
        login(token=args.token)
        print("Logged in with provided token")
    else:
        print("Please run 'huggingface-cli login' if not already logged in")
    
    # 获取用户名
    username = args.username
    if not username:
        username = input("Enter your HuggingFace username: ")
    
    api = HfApi()
    
    for repo_name in args.repos:
        full_repo_name = f"{username}/{repo_name}"
        local_dir = repo_name
        
        # 创建仓库
        print(f"\nCreating repository: {full_repo_name}...")
        create_repo(full_repo_name, repo_type="dataset", exist_ok=True)
        
        # 上传文件夹
        print(f"Uploading {local_dir} to {full_repo_name}...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=full_repo_name,
            repo_type="dataset"
        )
        
        print(f"Successfully uploaded {local_dir} to {full_repo_name}")
    
    print("\nAll uploads completed!")

if __name__ == "__main__":
    main()