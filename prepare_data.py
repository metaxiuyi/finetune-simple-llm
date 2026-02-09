import json
from datasets import load_dataset
import os

def format_data(example):
    # Try to handle common column names for instruction/output
    instruction = example.get('instruction', '') or example.get('question', '') or example.get('input', '')
    output = example.get('output', '') or example.get('answer', '') or example.get('response', '')
    
    if not instruction or not output:
        return None
        
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }

def main():
    print("Loading dataset 'LooksJuicy/ruozhiba'...")
    try:
        # Trying a known popular ruozhiba dataset source
        # If this specific one fails, user might need to adjust or provide a local file
        ds = load_dataset("LooksJuicy/ruozhiba", split="train")
    except Exception as e:
        print(f"Error loading primary dataset: {e}")
        print("Trying fallback 'IreneZ/ruozhiba_gpt4_turbo'...")
        try:
            ds = load_dataset("IreneZ/ruozhiba_gpt4_turbo", split="train")
        except Exception as e2:
            print(f"Error loading fallback dataset: {e2}")
            print("Please ensure you have internet access or provide a local dataset.")
            return

    print(f"Dataset loaded. Size: {len(ds)}")
    
    output_file = "ruozhiba_formatted.jsonl"
    print(f"Formatting and saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in ds:
            formatted = format_data(item)
            if formatted:
                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
                
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
