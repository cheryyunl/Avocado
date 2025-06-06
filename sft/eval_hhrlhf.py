import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding
from peft import PeftModel, PeftConfig
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, check_lora_in_model_path, Instructions, Instructions_summary, \
                   build_dataset_eval, build_dataset_summary_eval, get_clean_data
tqdm.pandas()

# 定义数据集路径
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'


@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/sft/logs_trl/avocado/sft_famo_0.5')
    dpo_model_path: Optional[str] = field(default='/cmlscratch/cheryunl/Avocado/dpo/output/dev/dpo/best_checkpoint')
    wandb_name: Optional[str] = field(default='eval_dpo_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names: Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name = script_args.base_model_name
dpo_model_path = script_args.dpo_model_path
tokenizer_name = script_args.base_model_name

model_path = dpo_model_path if (dpo_model_path is not None and os.path.exists(dpo_model_path)) else base_model_name
model_type = "DPO" if model_path == dpo_model_path else "SFT"
print(f"Using {model_type} model for evaluation: {model_path}")

process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(reward_names)
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])

reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

set_seed(8888)
tokenizer = load_main_tokenizer(tokenizer_name)

if os.path.exists(os.path.join(model_path, "adapter_config.json")):
    print("Loading model as PEFT adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    peft_config = PeftConfig.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id,
    )
    model.resize_token_embeddings(len(tokenizer))

if hasattr(model, 'merge_and_unload'):
    print("Merging and unloading adapters...")
    model = model.merge_and_unload()

generation_kwargs = {
    "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9, 
    "do_sample": True,
}

print('Evaluating DPO model...')
tokenizer.padding_side = "right"

if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers[0], reward_models.rm_tokenizers[1], split='test') 
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

valid_batch_size = 1
remove_keys = []
for key in ['key', 'text', 'prompt', 'response', 'query']:
    if key in valid_dataset.column_names:
        remove_keys.append(key)
valid_dataset = valid_dataset.remove_columns(remove_keys)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, drop_last=True, collate_fn=data_collator)
accelerator = Accelerator()
model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

full_response_tensors = []
full_prompts = []

pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
with torch.no_grad():
    for i, batch in enumerate(valid_data_loader):
        response_tensors = accelerator.unwrap_model(model).generate(batch['input_ids'], attention_mask=batch['attention_mask'], **generation_kwargs)
        full_response_tensors.extend(response_tensors)
        full_prompts.extend(batch['input_ids'])
        pbar.update(1)

full_prompts = tokenizer.batch_decode(full_prompts)
full_responses = tokenizer.batch_decode(full_response_tensors)
full_responses = get_clean_data(full_responses, full_prompts)

queries_responses = [
    (instructions.get_input(text), instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)

# 合并数据
all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
all_full_prompts = accelerator.gather_for_metrics(full_prompts)
all_full_responses = accelerator.gather_for_metrics(full_responses)

if process_id == 0:
    evaluation_result = {
        'prompt': all_full_prompts,
        'response': all_full_responses,
    }
    
    # 添加DPO模型的得分
    for i in range(reward_models.num_rewards):
        evaluation_result[f'score_{reward_names[i]}'] = all_rewards[i]
        print(f'DPO model average {reward_names[i]} score: {np.mean(all_rewards[i]):.4f}')

    # 保存结果
    dataframe = pd.DataFrame(evaluation_result)
    dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name, 'dpo_eval_results.csv'))