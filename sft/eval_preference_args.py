import os
import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any, Dict, Tuple
import torch
from torch.nn import functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from trl import set_seed
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入您原始评估代码中的工具函数
from multi_reward_models import RewardModels
from utils import load_main_tokenizer, check_lora_in_model_path, Instructions, Instructions_summary, \
                    build_dataset_eval, build_dataset_summary_eval, get_clean_data

# ARGS实现 - 从您提供的代码中提取
def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]

# 适配器类，连接ARGS和您的评估代码
class ARGSAdapter:
    def __init__(
        self, 
        model,
        tokenizer,
        reward_models,
        device="cuda",
        rm_device="cuda",
        debug=False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rm_models = reward_models.reward_models  # 获取原始reward models列表
        self.rm_tokenizers = reward_models.rm_tokenizers
        self.num_rewards = reward_models.num_rewards
        self.device = device
        self.rm_device = rm_device
        self.debug = debug

        for i, rm_tokenizer in enumerate(self.rm_tokenizers):
            if rm_tokenizer.pad_token is None:
                rm_tokenizer.pad_token = rm_tokenizer.eos_token
                rm_tokenizer.padding_side = 'right'
        
    def generate_step(self, mout, input_ids, pre_screen_beam_width=20, weights=None, temperature=0.7, rm_cached=None):
        if weights is None:
            weights = [1.0/self.num_rewards] * self.num_rewards
            
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        
        # 文本解码用于reward评估
        candidate_texts = self.tokenizer.batch_decode(flat_trme, skip_special_tokens=True)
        
        # 使用多个reward models评估
        all_rewards = torch.zeros(flat_trme.shape[0], device=self.device)
        for i, (rm, rm_tokenizer, weight) in enumerate(zip(self.rm_models, self.rm_tokenizers, weights)):
            # 为reward model准备输入
            rm_inputs = rm_tokenizer(candidate_texts, return_tensors="pt", padding=True, truncation=True).to(self.rm_device)
            with torch.no_grad():
                rewards = rm(**rm_inputs).logits.flatten().to(self.device)
            
            # 加权合并reward
            all_rewards += weight * rewards
            
        # 合并语言模型logits和reward分数
        new_scores = all_rewards * sum(weights) + prescreen_logits.flatten()
        
        if temperature > 0:
            # 使用temperature sampling
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
        else:
            # 使用greedy选择
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            
        return flat_trme[top_k_ids], None  # 不传递rm_cached因为我们使用解码+评分
        
    def args_generate(
        self, 
        input_ids, 
        attention_mask, 
        instructions,
        preference_weights=None,
        beta=0.5,   # reward影响系数
        topk=20,    # 考虑的候选token数量
        max_new_tokens=128,
        temperature=0.7,
        **generation_kwargs
    ):
        """使用ARGS进行生成"""
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 初始化生成序列
        curr_input_ids = input_ids.clone()
        curr_attention_mask = attention_mask.clone()
        
        # 跟踪未完成的序列
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # 使用模型缓存提高效率
        cached = None
        
        # 生成tokens
        for _ in range(max_new_tokens):
            # 如果所有序列都已完成，则停止
            if not unfinished.any():
                break
                
            # 获取模型输出
            with torch.no_grad():
                if cached is None:
                    mout = self.model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        use_cache=True
                    )
                    cached = mout.past_key_values
                else:
                    mout = self.model(
                        input_ids=curr_input_ids[:, -1].unsqueeze(-1),
                        attention_mask=curr_attention_mask,
                        past_key_values=cached,
                        use_cache=True
                    )
                    cached = mout.past_key_values
                
                # 使用ARGS步骤生成
                next_tokens, _ = self.generate_step(
                    mout, 
                    curr_input_ids, 
                    pre_screen_beam_width=topk,
                    weights=preference_weights,
                    temperature=temperature
                )
                
                # 更新input_ids
                curr_input_ids = next_tokens
                
                # 更新attention_mask
                curr_attention_mask = torch.cat([
                    curr_attention_mask, 
                    torch.ones((batch_size, 1), dtype=torch.long, device=device)
                ], dim=1)
                
                # 检查EOS是否生成
                eos_mask = (curr_input_ids[:, -1] == self.tokenizer.eos_token_id)
                unfinished = unfinished & ~eos_mask
        
        return curr_input_ids

# 修改原始评估代码
@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_trl')
    base_model_name: Optional[str] = field(default='./huggingface_models/Llama-2-7b-hf')
    wandb_name: Optional[str] = field(default='evalnew_assistant_pretrained_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    
    # ARGS参数
    use_args: Optional[bool] = field(default=True, metadata={"help": "whether to use ARGS for generation"})
    beta: Optional[float] = field(default=0.5, metadata={"help": "beta parameter for reward influence"})
    topk: Optional[int] = field(default=20, metadata={"help": "topk parameter for candidate tokens"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "temperature for sampling"})
    preference_weights: Optional[str] = field(default="0.5,0.5", metadata={"help": "comma-separated weights for reward models"})

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    exp_type = script_args.exp_type
    base_model_name = script_args.base_model_name
    tokenizer_name = script_args.base_model_name
    print('base model: ', base_model_name)

    # 设置GPU
    from accelerate import Accelerator
    process_id = Accelerator().local_process_index 
    gpu_id = process_id 
    print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

    # 解析reward模型
    reward_names = [x.strip() for x in script_args.reward_names.split(',')]
    preference_weights = [float(x.strip()) for x in script_args.preference_weights.split(',')]
    print('Reward models: ', reward_names)
    print('Preference weights: ', preference_weights)

    # 加载reward模型
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

    # 设置随机种子
    set_seed(8888)
    
    # 加载语言模型和tokenizer
    tokenizer = load_main_tokenizer(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id, 
    )
    model.resize_token_embeddings(len(tokenizer))
    if check_lora_in_model_path(model, base_model_name):
        model = PeftModel.from_pretrained(model, base_model_name)
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()

    # 设置生成参数
    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }

    # 准备评估数据集
    print('evaluation........')
    tokenizer.padding_side = "left"

    if exp_type == 'assistant':
        valid_dataset = build_dataset_eval(
            'Anthropic/hh-rlhf', 
            tokenizer, 
            reward_models.rm_tokenizers[0], 
            reward_models.rm_tokenizers[1], 
            split='test'
        ) 
        instructions = Instructions()
    else:
        valid_dataset = build_dataset_summary_eval(
            'openai/summarize_from_feedback', 
            tokenizer, 
            reward_models.rm_tokenizers[0], 
            reward_models.rm_tokenizers[1], 
            split='test'
        ) 
        instructions = Instructions_summary()
    print(f"Size of the validation set: {len(valid_dataset)}")

    # 准备数据加载器
    valid_batch_size = 1
    remove_keys = []
    for key in ['key', 'text', 'prompt', 'response', 'query']:
        if key in valid_dataset.column_names:
            remove_keys.append(key)
    valid_dataset = valid_dataset.remove_columns(remove_keys)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=valid_batch_size, 
        drop_last=True, 
        collate_fn=data_collator
    )

    accelerator = Accelerator()
    model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    if script_args.use_args:
        args_adapter = ARGSAdapter(
            model=accelerator.unwrap_model(model),
            tokenizer=tokenizer,
            reward_models=reward_models,
            device=gpu_id,
            rm_device=gpu_id
        )

    # 评估循环
    full_response_tensors = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // valid_batch_size // accelerator.num_processes)
    
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            # 根据选择的方法进行生成
            if script_args.use_args:
                # 使用ARGS生成
                response_tensors = args_adapter.args_generate(
                    batch['input_ids'],
                    batch['attention_mask'],
                    instructions,
                    preference_weights=preference_weights,
                    beta=script_args.beta,
                    topk=script_args.topk,
                    temperature=script_args.temperature,
                    **generation_kwargs
                )
            else:
                # 使用标准生成
                response_tensors = accelerator.unwrap_model(model).generate(
                    batch['input_ids'], 
                    attention_mask=batch['attention_mask'], 
                    **generation_kwargs
                )
                
            full_response_tensors.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)

    # 解码生成的文本
    full_prompts = tokenizer.batch_decode(full_prompts)
    full_responses = tokenizer.batch_decode(full_response_tensors)
    full_responses = get_clean_data(full_responses, full_prompts)
    
    # 计算reward分数
    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]

    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)

    # 收集结果
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)

    # 保存结果
    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        # 在结果文件名中添加生成方法信息
        if script_args.use_args:
            filename = os.path.join(
                script_args.save_directory, 
                script_args.wandb_name,
                f'eval_data_ARGS_beta{script_args.beta}_topk{script_args.topk}.csv'
            )
        else:
            filename = os.path.join(
                script_args.save_directory, 
                script_args.wandb_name,
                'eval_data.csv'
            )
            
        dataframe = pd.DataFrame(evaluation_result)
        dataframe.to_csv(filename)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    main()