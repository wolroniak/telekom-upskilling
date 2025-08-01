
import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def fine_tune_model(
    dataset_path="data/finetuning_dataset",
    base_model_id="Qwen/Qwen3-0.6B",
    output_dir="models/Qwen3-0.6B-fine-tuned",
    lora_rank=64,
    lora_alpha=16,
    lora_dropout=0.05,
    learning_rate=2e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    use_4bit=True,
):
    """
    Fine-tunes the Qwen3 model using the modern SFTTrainer API.
    """
    # --- Load the dataset ---
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run the data_preparation.py script first.")
    dataset = load_from_disk(dataset_path)

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # --- Correctly prepare model for k-bit training ---
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # --- Configure LoRA ---
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)

    # --- Data Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Set up training arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=25,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="tensorboard",
    )

    # --- Initialize Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    # --- Start Training ---
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")

    # --- Save the final model ---
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model adapter saved to {final_model_path}")

if __name__ == "__main__":
    fine_tune_model()
