# dpo_idefics2-8b.py
from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


def main():
    # Load the model and processor
    model_path = "/nfs/turbo/coe-chaijy/xuejunzh/idefics2-8b"  # 修改为你存储模型的实际路径
    model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_path, do_image_splitting=False)


    # Load the dataset
    cache_dir = "/nfs/turbo/coe-chaijy/xuejunzh/TRLAIF-V-Dataset"

    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train", cache_dir=cache_dir)   

    def format(example):
        # Prepare the input for the chat template
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        # Apply the chat template
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)
        # Resize the image to ensure it fits within the maximum allowable
        # size of the processor to prevent OOM errors.
        max_size = processor.image_processor.size["longest_edge"] // 2
        example["image"].thumbnail((max_size, max_size))
        return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

    # Apply the formatting function to the dataset
    dataset = dataset.map(format, remove_columns=dataset.column_names, num_proc=32)

    # Make sure that the images are decoded, it prevents from storing bytes.
    # More info here https://github.com/huggingface/blog/pull/2148#discussion_r1667400478
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    # Train the model
    training_args = DPOConfig(
        output_dir="idefics2-8b-dpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        dataset_num_proc=8,  # tokenization will use 32 processes
        dataloader_num_workers=8,  # data loading will use 32 workers
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,  # not needed when using peft
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        peft_config=LoraConfig(target_modules="all-linear"),
    )

    trainer.train()


if __name__ == "__main__":
    main()
