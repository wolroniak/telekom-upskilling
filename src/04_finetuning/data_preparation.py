
import os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def create_finetuning_dataset(
    alpaca_samples=3000, 
    hh_rlhf_samples=3000, 
    output_dir="data",
    model_id="Qwen/Qwen3-0.6B"
):
    """
    Creates a combined fine-tuning dataset from Alpaca and HH-RLHF.
    Ensures each example is a single text block ending with the EOS token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    EOS_TOKEN = tokenizer.eos_token
    if not EOS_TOKEN:
        raise ValueError("Tokenizer must have an EOS token.")

    # --- Formatting functions ---
    def format_alpaca(example):
        """Formats an Alpaca example into a single text block ending with EOS."""
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        if not text.endswith(EOS_TOKEN):
            text += EOS_TOKEN
        return {"text": text}

    def format_hh_rlhf(example):
        """Formats an HH-RLHF example into a single text block ending with EOS."""
        text = example['chosen']
        if not text.endswith(EOS_TOKEN):
            text += EOS_TOKEN
        return {"text": text}

    # --- Load and process Alpaca dataset ---
    print("Processing Alpaca dataset...")
    alpaca_dataset = load_dataset("pankajmathur/alpaca_orca", split="train")
    if alpaca_samples > 0:
        alpaca_dataset = alpaca_dataset.select(range(alpaca_samples))
    
    alpaca_formatted = alpaca_dataset.map(format_alpaca, remove_columns=[col for col in alpaca_dataset.column_names if col != 'text'])

    # --- Load and process Anthropic HH-RLHF dataset ---
    print("Processing Anthropic HH-RLHF dataset...")
    hh_rlhf_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    if hh_rlhf_samples > 0:
        hh_rlhf_dataset = hh_rlhf_dataset.select(range(hh_rlhf_samples))
        
    hh_rlhf_formatted = hh_rlhf_dataset.map(format_hh_rlhf, remove_columns=[col for col in hh_rlhf_dataset.column_names if col != 'text'])

    # --- Combine and save ---
    print("Combining and shuffling datasets...")
    combined_dataset = concatenate_datasets([alpaca_formatted, hh_rlhf_formatted]).shuffle(seed=42)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_path = os.path.join(output_dir, "finetuning_dataset")
    combined_dataset.save_to_disk(dataset_path)
    
    print(f"Successfully created and saved the dataset to {dataset_path}")
    print(f"Total examples: {len(combined_dataset)}")
    print("\nExample entry from the new dataset:\n", combined_dataset[0]['text'])

if __name__ == "__main__":
    create_finetuning_dataset()
