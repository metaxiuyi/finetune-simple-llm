import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    data_file = "ruozhiba_formatted.jsonl"
    output_dir = "qwen_ruozhiba_finetuned"
    
    # 1. Load Dataset
    print(f"Loading data from {data_file}...")
    try:
        dataset = load_dataset("json", data_files=data_file, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please run prepare_data.py first!")
        return

    # 2. Load Tokenizer & Model
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Optional: Quantization configuration for lower memory usage
    # Since 1.5B is small, we might not strictly need 4bit, but it helps on smaller GPUs.
    # We'll use 4-bit quantization by default for broader compatibility.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 3. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        max_seq_length=1024,
        packing=False, # Use standard packing
        dataset_text_field=None, # We use messages
    )
    
    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Save
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
