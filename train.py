import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    data_file = "ruozhiba_formatted.jsonl"
    output_dir = "qwen_ruozhiba_finetuned"
    max_seq_length = 1024
    
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
    
    # Qwen2.5 tokenizer usually has eos_token. We ensure pad_token is set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization Config
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Data Preprocessing (Native implementation without trl)
    def preprocess_function(examples):
        inputs = []
        for messages in examples["messages"]:
            # Apply chat template to format the conversation
            # tokenize=False returns a string
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs.append(text)
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            truncation=True,
            padding=False, # We will pad in data_collator
        )
        
        # For Causal LM, labels are typically the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False, # Important for custom dataset columns if any, though we removed them
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        # DataCollatorForSeq2Seq handles padding and label masking (ignoring pad tokens)
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
