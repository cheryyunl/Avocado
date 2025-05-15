import os
import pandas as pd
import numpy as np
from glob import glob

def merge_gpu_results(base_dir, wandb_name):
    """合并所有GPU的结果文件"""
    # 确保base_dir是绝对路径
    base_dir = os.path.abspath(base_dir)
    output_dir = os.path.join(base_dir, wandb_name)
    
    print(f"Looking for GPU results in: {output_dir}")
    
    # 查找所有gpu_*目录
    gpu_dirs = [d for d in glob(os.path.join(output_dir, "gpu_*")) if os.path.isdir(d)]
    
    if not gpu_dirs:
        print(f"No GPU result directories found in {output_dir}")
        print("Available directories:")
        for item in os.listdir(output_dir):
            print(f"  - {item}")
        return
    
    print(f"Found GPU directories: {gpu_dirs}")
    
    # 获取所有权重组合
    weight_patterns = set()
    for gpu_dir in gpu_dirs:
        files = glob(os.path.join(gpu_dir, "helpsteer_eval_*.csv"))
        for file in files:
            if "weights" in file:
                # 提取权重字符串 - 修复提取方式
                try:
                    # 从文件名中提取权重部分
                    weight_part = file.split("weights")[-1].split(".")[0]
                    # 确保权重格式正确
                    if weight_part.startswith("-"):
                        weight_part = weight_part[1:]
                    if weight_part.endswith("-"):
                        weight_part = weight_part[:-1]
                    weight_patterns.add(weight_part)
                    print(f"Extracted weight pattern: {weight_part} from {file}")
                except Exception as e:
                    print(f"Error extracting weight pattern from {file}: {str(e)}")
    
    if not weight_patterns:
        print("No weight patterns found in the results")
        return
    
    print(f"Found weight patterns: {weight_patterns}")
    
    # 对每个权重组合合并结果
    for weight_str in weight_patterns:
        print(f"\nProcessing weight combination: {weight_str}")
        all_results = []
        
        # 收集所有GPU的结果
        for gpu_dir in gpu_dirs:
            # 修改文件匹配模式
            pattern = f"*weights{weight_str}.csv"
            if not weight_str.startswith("-"):
                pattern = f"*weights-{weight_str}.csv"
            files = glob(os.path.join(gpu_dir, pattern))
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    all_results.append(df)
                    print(f"Loaded results from {file}")
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
        
        if not all_results:
            print(f"No results found for weights {weight_str}")
            continue
        
        # 合并所有结果
        merged_df = pd.concat(all_results, ignore_index=True)
        
        # 计算平均分数
        attributes = [col for col in merged_df.columns if col.startswith('helpsteer-')]
        summary = {
            'total_samples': len(merged_df),
        }
        
        for attr in attributes:
            summary[f'avg_{attr}'] = merged_df[attr].mean()
        
        summary['avg_overall_score'] = merged_df['overall_score'].mean()
        
        # 保存合并后的结果
        merged_filename = os.path.join(output_dir, f"merged_results_weights{weight_str}.csv")
        merged_df.to_csv(merged_filename, index=False)
        print(f"Saved merged results to {merged_filename}")
        
        # 打印汇总信息
        print("\nSummary:")
        print(f"Total samples: {summary['total_samples']}")
        for attr in attributes:
            print(f"Average {attr}: {summary[f'avg_{attr}']:.4f}")
        print(f"Average overall score: {summary['avg_overall_score']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing the results")
    parser.add_argument("--wandb_name", type=str, required=True, help="Name of the wandb run")
    args = parser.parse_args()
    
    merge_gpu_results(args.base_dir, args.wandb_name) 