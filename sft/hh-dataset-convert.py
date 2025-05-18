from datasets import load_dataset, concatenate_datasets, Dataset
import re
import pandas as pd
import os
import sys

def convert_anthropic_to_dahoas_format(examples):
    """
    将Anthropic的HH-RLHF数据集格式转换为Dahoas的full-hh-rlhf格式
    
    Anthropic格式: 'chosen', 'rejected' 包含完整的多轮Human/Assistant对话
    Dahoas格式: 
        - prompt: 包含完整对话历史（包括所有Human:和Assistant:前缀），包含最后一个"Assistant:"前缀，但不包含最后的回答
        - response: 与chosen相同，不包含"Assistant:"前缀的被选择回答
        - chosen: 不包含"Assistant:"前缀的被选择回答
        - rejected: 不包含"Assistant:"前缀的被拒绝回答
    """
    new_examples = {
        "prompt": [],
        "response": [],
        "chosen": [],
        "rejected": []
    }
    
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        # 确保输入数据有效
        if not (chosen and rejected):
            continue
            
        try:
            # 使用正则表达式找到最后一个Assistant:
            last_human_pos = chosen.rfind("Human: ")
            if last_human_pos == -1:
                continue
                
            # 找到最后一个Human:之后的Assistant:
            last_assistant_pos = chosen.find("Assistant:", last_human_pos)
            if last_assistant_pos == -1:
                continue
                
            # 提取回答内容（Assistant:之后的部分）
            answer_start = last_assistant_pos + len("Assistant:")
            chosen_answer = chosen[answer_start:].strip()
            
            # 构建prompt（所有内容直到并包括"Assistant:"）
            prompt = chosen[:answer_start].strip()
            
            # 从rejected提取回答
            last_rejected_human_pos = rejected.rfind("Human: ")
            if last_rejected_human_pos == -1:
                continue
                
            last_rejected_assistant_pos = rejected.find("Assistant:", last_rejected_human_pos)
            if last_rejected_assistant_pos == -1:
                continue
                
            rejected_answer_start = last_rejected_assistant_pos + len("Assistant:")
            rejected_answer = rejected[rejected_answer_start:].strip()
            
            new_examples["prompt"].append(prompt)
            new_examples["response"].append(chosen_answer)
            new_examples["chosen"].append(chosen_answer)
            new_examples["rejected"].append(rejected_answer)
            
        except Exception as e:
            print(f"Error processing example with first method: {e}")
            # 尝试另一种解析方法
            try:
                # 使用更简单的字符串查找方法
                parts = chosen.split("Assistant:")
                if len(parts) < 2:
                    continue
                    
                # 最后一个回答
                last_answer = parts[-1].strip()
                
                # prompt是包括最后一个"Assistant:"在内的所有之前内容
                prompt = "Assistant:".join(parts[:-1]) + "Assistant:"
                
                # 处理rejected
                rejected_parts = rejected.split("Assistant:")
                if len(rejected_parts) < 2:
                    continue
                    
                rejected_answer = rejected_parts[-1].strip()
                
                new_examples["prompt"].append(prompt)
                new_examples["response"].append(last_answer)
                new_examples["chosen"].append(last_answer)
                new_examples["rejected"].append(rejected_answer)
                
            except Exception as e2:
                print(f"Error processing example with second method: {e2}")
                continue
    
    return new_examples
    
    return new_examples
    
    return new_examples
    
    return new_examples

def process_dataset(dataset_name, data_dir, split="train"):
    """加载并处理指定的数据集"""
    print(f"Processing {dataset_name}/{data_dir}...")
    ds = load_dataset(dataset_name, data_dir=data_dir, split=split)
    
    # 将数据集转换为Dahoas格式
    converted_data = convert_anthropic_to_dahoas_format(ds)
    
    # 创建新的数据集
    return Dataset.from_dict(converted_data)

def main():
    # 设置基本参数
    dataset_name = "Anthropic/hh-rlhf"
    split = "train"  # 或者 "test"，根据需要
    
    # 测试转换函数
    print("Testing conversion function with a sample...")
    test_sample()
    
    # 加载并处理harmless数据集
    print("\nProcessing harmless datasets...")
    ds_harmless = process_dataset(dataset_name, "harmless-base", split)
    
    # 加载并处理red-team-attempts数据集（如果可用）
    try:
        print("Processing red-team-attempts...")
        ds_red_team = load_dataset(dataset_name, data_dir="red-team-attempts", split=split)
        
        # 转换red-team-attempts数据
        converted_red_team = convert_anthropic_to_dahoas_format(ds_red_team)
        ds_red_team_converted = Dataset.from_dict(converted_red_team)
        
        # 将red-team-attempts添加到harmless数据集
        ds_harmless = concatenate_datasets([ds_harmless, ds_red_team_converted])
        print(f"Added {len(ds_red_team_converted)} examples from red-team-attempts to harmless dataset")
    except Exception as e:
        print(f"Error processing red-team-attempts: {e}")
    
    # 加载并处理helpful数据集
    print("Processing helpful datasets...")
    ds_helpful_base = process_dataset(dataset_name, "helpful-base", split)
    ds_helpful_online = process_dataset(dataset_name, "helpful-online", split)
    ds_helpful_rs = process_dataset(dataset_name, "helpful-rejection-sampled", split)
    
    # 合并所有helpful数据集
    ds_helpful = concatenate_datasets([ds_helpful_base, ds_helpful_online, ds_helpful_rs])
    
    # 显示结果统计
    print(f"\nFinal dataset statistics:")
    print(f"Harmless dataset: {len(ds_harmless)} examples")
    print(f"Helpful dataset: {len(ds_helpful)} examples")
    
    # 保存转换后的数据集到本地
    print("\nSaving datasets to disk...")
    ds_harmless.save_to_disk("converted_harmless")
    ds_helpful.save_to_disk("converted_helpful")
    
    # 询问是否上传到HuggingFace Hub
    print("\nDo you want to upload the datasets to HuggingFace Hub? (y/n)")
    answer = input().strip().lower()
    
    if answer == 'y':
        # 导入所需的登录函数
        try:
            from huggingface_hub import login
            
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
            
            # 询问数据集名称
            print("\nPlease enter a name for the harmless dataset (default: converted-hh-rlhf-harmless):")
            harmless_name = input().strip() or "converted-hh-rlhf-harmless"
            
            print("\nPlease enter a name for the helpful dataset (default: converted-hh-rlhf-helpful):")
            helpful_name = input().strip() or "converted-hh-rlhf-helpful"
            
            # 上传数据集
            print(f"\nUploading harmless dataset to {username}/{harmless_name}...")
            ds_harmless.push_to_hub(f"{username}/{harmless_name}")
            
            print(f"\nUploading helpful dataset to {username}/{helpful_name}...")
            ds_helpful.push_to_hub(f"{username}/{helpful_name}")
            
            print("\nDatasets uploaded successfully!")
            
        except ImportError:
            print("huggingface_hub package is not installed. Please install it with:")
            print("pip install huggingface_hub")
            
        except Exception as e:
            print(f"Error during upload: {e}")
    
    print("\nConversion completed successfully!")

def test_sample():
    """测试转换函数，使用您提供的示例和额外的测试案例"""
    
    # 来自您提供的示例
    sample1 = {
        "chosen": [
            "Human: What are some cuss words in english? Assistant: Here's an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama's boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: I haven't even thought about it."
        ],
        "rejected": [
            "Human: What are some cuss words in english? Assistant: Here's an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama's boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one? Assistant: Ass."
        ]
    }
    
    # 另一个更简单的测试案例
    sample2 = {
        "chosen": [
            "Human: How are you today? Assistant: I'm doing well, thank you for asking! How can I help you? Human: Tell me a joke. Assistant: Why don't scientists trust atoms? Because they make up everything!"
        ],
        "rejected": [
            "Human: How are you today? Assistant: I'm doing well, thank you for asking! How can I help you? Human: Tell me a joke. Assistant: Your life."
        ]
    }
    
    # 测试第一个样本
    print("\n=== Testing Sample 1 ===")
    result1 = convert_anthropic_to_dahoas_format(sample1)
    
    print("Original sample:")
    print(f"Chosen: {sample1['chosen'][0]}")
    print(f"Rejected: {sample1['rejected'][0]}")
    print("\nConverted sample:")
    print(f"Prompt: {result1['prompt'][0]}")
    print(f"Response: {result1['response'][0]}")
    print(f"Chosen: {result1['chosen'][0]}")
    print(f"Rejected: {result1['rejected'][0]}")
    
    # 验证结果是否符合预期
    expected_prompt1 = "Human: What are some cuss words in english? Assistant: Here's an incomplete list. Ass, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama's boy, faggot, pervert, queer, scumbag, bitch, Human: What's your favorite one?"
    expected_response1 = "Assistant: I haven't even thought about it."
    expected_chosen1 = "I haven't even thought about it."
    expected_rejected1 = "Ass."
    
    print("\nValidation:")
    print(f"Prompt matches expected: {result1['prompt'][0] == expected_prompt1}")
    print(f"Response matches expected: {result1['response'][0] == expected_response1}")
    print(f"Chosen matches expected: {result1['chosen'][0] == expected_chosen1}")
    print(f"Rejected matches expected: {result1['rejected'][0] == expected_rejected1}")
    
    # 测试第二个样本
    print("\n=== Testing Sample 2 ===")
    result2 = convert_anthropic_to_dahoas_format(sample2)
    
    print("Original sample:")
    print(f"Chosen: {sample2['chosen'][0]}")
    print(f"Rejected: {sample2['rejected'][0]}")
    print("\nConverted sample:")
    print(f"Prompt: {result2['prompt'][0]}")
    print(f"Response: {result2['response'][0]}")
    print(f"Chosen: {result2['chosen'][0]}")
    print(f"Rejected: {result2['rejected'][0]}")
    
    # 验证结果是否符合预期
    expected_prompt2 = "Human: How are you today? Assistant: I'm doing well, thank you for asking! How can I help you? Human: Tell me a joke."
    expected_response2 = "Assistant: Why don't scientists trust atoms? Because they make up everything!"
    expected_chosen2 = "Why don't scientists trust atoms? Because they make up everything!"
    expected_rejected2 = "Your life."
    
    print("\nValidation:")
    print(f"Prompt matches expected: {result2['prompt'][0] == expected_prompt2}")
    print(f"Response matches expected: {result2['response'][0] == expected_response2}")
    print(f"Chosen matches expected: {result2['chosen'][0] == expected_chosen2}")
    print(f"Rejected matches expected: {result2['rejected'][0] == expected_rejected2}")
    
    # 来自您第二个例子的样本
    sample3 = {
        "chosen": [
            "Human: Should you buy a case to protect your cell phone? Assistant: It depends on your circumstances. If you carry your phone in a pocket or a purse then you probably want a case. But if you only need a phone for quick interactions, a case may actually cause more harm than good. What do you need the phone for? Are you a parent, or do you work from home? Human: What harm could it do? Assistant: A phone case can damage the screen, for one thing. It can also get you in trouble if you have your phone turned off for some reason. Then you will turn it back on and it won't do anything. If you can afford to replace it, then you need a case to protect it. The problem is that most people aren't able to afford to replace their phones all the time. Human: Thanks for letting me know. Assistant: You're welcome."
        ],
        "rejected": [
            "Human: Should you buy a case to protect your cell phone? Assistant: It depends on your circumstances. If you carry your phone in a pocket or a purse then you probably want a case. But if you only need a phone for quick interactions, a case may actually cause more harm than good. What do you need the phone for? Are you a parent, or do you work from home? Human: What harm could it do? Assistant: A phone case can damage the screen, for one thing. It can also get you in trouble if you have your phone turned off for some reason. Then you will turn it back on and it won't do anything. If you can afford to replace it, then you need a case to protect it. The problem is that most people aren't able to afford to replace their phones all the time. Human: Thanks for letting me know. Assistant: You're welcome.\nYou're welcome."
        ]
    }
    
    # 测试第三个样本
    print("\n=== Testing Sample 3 ===")
    result3 = convert_anthropic_to_dahoas_format(sample3)
    
    print("Original sample:")
    print(f"Chosen: {sample3['chosen'][0]}")
    print(f"Rejected: {sample3['rejected'][0]}")
    print("\nConverted sample:")
    print(f"Prompt: {result3['prompt'][0]}")
    print(f"Response: {result3['response'][0]}")
    print(f"Chosen: {result3['chosen'][0]}")
    print(f"Rejected: {result3['rejected'][0]}")
    
    expected_prompt3 = "Human: Should you buy a case to protect your cell phone? Assistant: It depends on your circumstances. If you carry your phone in a pocket or a purse then you probably want a case. But if you only need a phone for quick interactions, a case may actually cause more harm than good. What do you need the phone for? Are you a parent, or do you work from home? Human: What harm could it do? Assistant: A phone case can damage the screen, for one thing. It can also get you in trouble if you have your phone turned off for some reason. Then you will turn it back on and it won't do anything. If you can afford to replace it, then you need a case to protect it. The problem is that most people aren't able to afford to replace their phones all the time. Human: Thanks for letting me know."
    expected_response3 = "Assistant: You're welcome."
    expected_chosen3 = "You're welcome."
    expected_rejected3 = "You're welcome.\nYou're welcome."
    
    print("\nValidation:")
    print(f"Prompt matches expected: {result3['prompt'][0] == expected_prompt3}")
    print(f"Response matches expected: {result3['response'][0] == expected_response3}")
    print(f"Chosen matches expected: {result3['chosen'][0] == expected_chosen3}")
    print(f"Rejected matches expected: {result3['rejected'][0] == expected_rejected3}")

    print("\nTest completed!")


if __name__ == "__main__":
    main()