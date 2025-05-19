import os
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
import numpy as np
import pandas as pd
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, Instructions, Instructions_summary
tqdm.pandas()

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    input_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to the jsonl file with pre-generated responses"})
    wandb_name: Optional[str] = field(default='eval_jsonl_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names: Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer to use"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
tokenizer_name = script_args.tokenizer_name

if script_args.input_jsonl is None or not os.path.exists(script_args.input_jsonl):
    raise ValueError(f"Input jsonl file not found: {script_args.input_jsonl}")

accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(f"Using reward models: {reward_names}")
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

tokenizer = load_main_tokenizer(tokenizer_name)

if script_args.input_jsonl.endswith('.jsonl'):
    with open(script_args.input_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
else: 
    with open(script_args.input_jsonl, 'r') as f:
        data = json.load(f)

print(f"loaded {len(data)} samples from {script_args.input_jsonl}")

if exp_type == 'assistant':
    instructions = Instructions()
else:
    instructions = Instructions_summary()

local_data = data[process_id::accelerator.num_processes]
print(f"process {process_id} evaluates {len(local_data)} samples")

full_responses = []
for item in local_data:
    full_responses.append(item['response'])

queries_responses = [
    (instructions.get_input(text), instructions.get_response(text))
    for text in full_responses
]

if hasattr(instructions, 'get_post'):
    rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
else:
    rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
all_rewards = []
for i in range(reward_models.num_rewards):
    all_rewards.append(accelerator.gather_for_metrics(torch.tensor(rewards_list[i])))

all_responses = accelerator.gather_for_metrics(full_responses)

if process_id == 0:
    methods = [item.get('method', '') for item in data]
    prompts = [item.get('prompt', '') for item in data]
    results = [item.get('result', '') for item in data]
    
    evaluation_result = {
        'prompt': prompts,
        'response': all_responses,
        'result': results,
        'method': methods
    }
    
    for i in range(reward_models.num_rewards):
        scores = all_rewards[i].tolist()
        evaluation_result[f'score_{reward_names[i]}'] = scores
        print(f'Average score of {reward_names[i]}: {np.mean(scores):.4f}')

    dataframe = pd.DataFrame(evaluation_result)
    output_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'jsonl_eval_results.csv')
    dataframe.to_csv(output_path)
    print(f"Results saved tos saved to {output_path}")