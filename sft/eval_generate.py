import os
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, Instructions, Instructions_summary
tqdm.pandas()

# HelpSteer属性名称
HELPSTEER_ATTRIBUTES = [
    'helpsteer-helpfulness',
    'helpsteer-correctness', 
    'helpsteer-coherence',
    'helpsteer-honesty',
    'helpsteer-complexity'
]

class HelpSteerRewardModel:
    def __init__(self, model_path="RLHFlow/RewardModel-Mistral-7B-for-DPA-v1", device="cuda:0"):
        print(f"Loading HelpSteer reward model from {model_path}...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'sliding_window') and config.sliding_window is None:
            config.sliding_window = 4096
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        
        # 输入模板
        self.input_template = "[INST] You must read the following conversation carefully and rate the assistant's response from score 0-100 in these aspects: helpfulness, correctness, coherence, honesty, complexity, verbosity\n\nUser: {prompt}\n\nAssistant: {response} [/INST]"
        
        self.num_rewards = len(HELPSTEER_ATTRIBUTES)
        print(f"Loaded HelpSteer model with {self.num_rewards} attributes")
        
    def get_reward_model_scores(self, query_response_pairs, get_post=None):
        """评估一批query-response对，返回每个属性的评分"""
        all_scores = [[] for _ in range(self.num_rewards)]
        
        for prompt, response in tqdm(query_response_pairs, desc="Evaluating with HelpSteer"):
            # 使用模板格式化输入
            formatted_input = self.input_template.format(prompt=prompt, response=response)
            
            # 编码输入
            model_inputs = self.tokenizer(
                formatted_input, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                raw_scores = self.model(**model_inputs).logits.squeeze().cpu().float().numpy()
                
                helpsteer_scores = (raw_scores[:self.num_rewards] - 10) / 20
                
                for i in range(self.num_rewards):
                    all_scores[i].append(float(helpsteer_scores[i]))
                    
                if len(all_scores[0]) == 1:
                    print(f"Raw scores: {raw_scores[:self.num_rewards]}")
                    print(f"HelpSteer scores: {helpsteer_scores}")
        
        return all_scores

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    input_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to the jsonl file with pre-generated responses"})
    wandb_name: Optional[str] = field(default='eval_jsonl_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names: Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Tokenizer to use"})
    helpsteer_model_path: Optional[str] = field(default='RLHFlow/RewardModel-Mistral-7B-for-DPA-v1', metadata={"help": "Path to HelpSteer reward model"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
tokenizer_name = script_args.tokenizer_name

if script_args.input_jsonl is None or not os.path.exists(script_args.input_jsonl):
    raise ValueError(f"Input jsonl file not found: {script_args.input_jsonl}")

accelerator = Accelerator()
process_id = accelerator.local_process_index 
gpu_id = process_id 
device = f"cuda:{gpu_id}"
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
print(f"Using reward models: {reward_names}")

# 检查是否使用helpsteer
if script_args.reward_names == 'helpsteer':
    # 使用HelpSteer reward model
    print("Using HelpSteer reward model")
    reward_models = HelpSteerRewardModel(script_args.helpsteer_model_path, device=device)
    reward_names = HELPSTEER_ATTRIBUTES
else:
    # 使用原来的reward models
    print("Using traditional reward models")
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

print(f"Reading file: {script_args.input_jsonl}")
with open(script_args.input_jsonl, 'r', encoding='utf-8') as f:
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
for i in range(len(rewards_list)):
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
    
    for i in range(len(rewards_list)):
        scores = all_rewards[i].tolist()
        evaluation_result[f'score_{reward_names[i]}'] = scores
        print(f'Average score of {reward_names[i]}: {np.mean(scores):.4f}')

    dataframe = pd.DataFrame(evaluation_result)
    output_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'jsonl_eval_results.csv')
    dataframe.to_csv(output_path)
    print(f"Results saved to {output_path}")