import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, HfArgumentParser, TrainingArguments
from trl import SFTTrainer, set_seed, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import numpy as np
import pandas as pd
import wandb
from accelerate import Accelerator
from multi_task_utils import build_mt_dataset, build_mt_reject_dataset, build_mt_all_dataset, build_harmless_dataset, build_helpful_dataset, MTDataCollatorForCompletionOnlyLM
from helpsteer_utils import build_helpsteer_mixed_dataset, build_helpsteer_positive_dataset, build_helpsteer_negative_dataset, HSDataCollatorForCompletionOnlyLM
from famo_trainer import FAMOSFTTrainer
from utils import load_main_tokenizer, Instructions_summary, build_dataset_summary, Instructions
from transformers import TrainerCallback
tqdm.pandas()

torch.cuda.empty_cache()

# os.environ['MASTER_PORT'] = '29501'

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'
helpsteer_dataset_path = 'cheryyunl/helpsteer'
# model_path = '/work/hdd/bcwu/cheryll/Llama-2-7b-hf'
model_path = '/cmlscratch/cheryunl/Llama-2-7b-hf'

class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'task_0_loss' in logs and 'task_2_loss' in logs:
            combined_loss = logs['task_0_loss'] + logs['task_2_loss']
            wandb.log({"multi_objective_loss": combined_loss}, step=state.global_step)

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    save_directory: Optional[str] = field(default='./logs_trl/')
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    wandb_name: Optional[str] = field(default='summary_sft_all_bs1_lora64', metadata={"help": "Name for this experiment"})
    exp_type: Optional[str] = field(default='summary', metadata={"help": "exp type, 'summary' or 'assistant' "})
    base_model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "local path to the base model or the huggingface id"})
    loss_scale: Optional[str] = field(default="{1: 1, 3: 0.4}", metadata={"help": "loss scale for objectives"})
    ema_alpha: Optional[float] = field(default=0.9, metadata={"help": "EMA smoothing factor"})
    init_steps: Optional[int] = field(default=100, metadata={"help": "Number of steps for EMA initialization"})
    w_lr: Optional[float] = field(default=1e-3, metadata={"help": "learning rate for w"})
    gamma: Optional[float] = field(default=0.01, metadata={"help": "gamma for famo"})
    famo_update_frequency: Optional[int] = field(default=10, metadata={"help": "famo update frequency"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
# base_model_name = script_args.base_model_name
base_model_name = model_path
tokenizer_name = base_model_name # we use the same tokenizer for the base model
print('base model: ', base_model_name)
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)

accelerator = Accelerator()

if accelerator.is_main_process:
    if script_args.log_with == 'wandb':
        wandb.init(
            project="Avocado",
            name=script_args.wandb_name,
            config=vars(script_args)
        )
else:
    wandb.init(mode="disabled")

if accelerator.is_main_process:
    if script_args.log_with == 'wandb':
        wandb.init(
            project="Avocado",
            name=script_args.wandb_name,
            config=vars(script_args)
        )
else:
    wandb.init(mode="disabled")

training_args = TrainingArguments(
    max_steps=20000,  
    output_dir=os.path.join(script_args.save_directory, script_args.wandb_name),
    dataloader_drop_last=True,
    eval_steps=30000,
    save_steps=10000,
    logging_steps=10,
    save_strategy='steps',
    per_device_train_batch_size=script_args.batch_size,
    per_device_eval_batch_size=script_args.batch_size,
    learning_rate=script_args.learning_rate,
    lr_scheduler_type="linear",
    warmup_steps=100,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=False,
    weight_decay=0.01,
    run_name=script_args.wandb_name,
    report_to='wandb' if (script_args.log_with == 'wandb' and accelerator.is_main_process) else 'none',
    ddp_find_unused_parameters=False,
    remove_unused_columns=True,
)

process_id = Accelerator().local_process_index 
gpu_id = process_id 
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))


# set seed before initializing value head for deterministic eval
set_seed(42)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = load_main_tokenizer(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
if script_args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        load_in_8bit=True, 
        device_map=gpu_id, 
        local_files_only=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.bfloat16,
        device_map=gpu_id, 
        local_files_only=True,
    )
model.resize_token_embeddings(len(tokenizer))

if exp_type == 'assistant':
    # dataset = build_dataset(hhrlhf_dataset_path, tokenizer, split='train') 
    # dataset = build_base_dataset(tokenizer, split='train') 
    train_dataset = build_mt_dataset(hhrlhf_dataset_path, tokenizer, split='train')  
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = MTDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    n_tasks = 2
    rejected_ids = None
elif exp_type == 'assistant_reject':
    train_dataset = build_mt_reject_dataset(hhrlhf_dataset_path, tokenizer, split='train')  
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = MTDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    rejected_ids = [0]
    n_tasks = 2
elif exp_type == 'assistant_all':
    train_dataset = build_mt_all_dataset(hhrlhf_dataset_path, tokenizer, split='train')  
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = MTDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    rejected_ids = [1, 3]
    n_tasks = 4
elif exp_type == 'assistant_harmless':
    train_dataset = build_harmless_dataset(hhrlhf_dataset_path, tokenizer, split='train')  
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = MTDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    rejected_ids = [1]
    n_tasks = 2
elif exp_type == 'assistant_helpful':
    train_dataset = build_helpful_dataset(hhrlhf_dataset_path, tokenizer, split='train')  
    response_template_ids = tokenizer.encode(Instructions.response_split, add_special_tokens=False)[1:]  
    collator = MTDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    rejected_ids = [1]
    n_tasks = 2
elif exp_type == 'helpsteer_positive':
    train_dataset = build_helpsteer_positive_dataset(helpsteer_dataset_path, tokenizer, split='train') 
    collator = HSDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    rejected_ids = None
    n_tasks = 4
elif exp_type == 'helpsteer_mixed':
    train_dataset = build_helpsteer_mixed_dataset(helpsteer_dataset_path, tokenizer, split='train') 
    collator = HSDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    rejected_ids = [3]
    n_tasks = 4
elif exp_type == 'helpsteer_negative':
    train_dataset = build_helpsteer_negative_dataset(helpsteer_dataset_path, tokenizer, split='train') 
    collator = HSDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    rejected_ids = [0,1,2,3]
    n_tasks = 4
else:
    dataset = build_dataset_summary(summary_dataset_path, tokenizer, split='train')
    response_template_ids = tokenizer.encode(Instructions_summary.response_split, add_special_tokens=False)[1:]  
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer, mlm=False)
    train_dataset = dataset.shuffle()

# print(f"Size of the train set: {len(train_dataset)}")

trainer = FAMOSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=lora_config,
    packing=False,
    dataset_text_field='query',
    data_collator=collator,
    n_tasks=n_tasks, 
    gamma=script_args.gamma,
    w_lr=script_args.w_lr,
    famo_update_frequency=script_args.famo_update_frequency,
    rejected_ids=rejected_ids,
    loss_scale=script_args.loss_scale,
    ema_alpha=script_args.ema_alpha,
    init_steps=script_args.init_steps,
)

trainer = accelerator.prepare(trainer)

# 添加回调
if script_args.log_with == 'wandb':
    trainer.add_callback(CustomCallback())

print("Training SFT model with multi-objective alignment")
trainer.train()

if accelerator.is_main_process:
    print("Saving last checkpoint of the model")
    save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'model')
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    if script_args.log_with == 'wandb':
        wandb.finish()
    

