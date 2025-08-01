
import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_complaints(file_path="src/01_prompt_engineering/complaints.json"):
    """Loads the customer complaints from the specified JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_response(model, tokenizer, system_prompt, user_prompt):
    """Generates a response from the given model and tokenizer using a chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False 
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, eos_token_id=tokenizer.eos_token_id)
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    parts = full_response.split('assistant')
    clean_response = parts[-1].strip() if len(parts) > 1 else full_response
    clean_response = clean_response.replace("<think>", "").replace("</think>", "").strip()
    return clean_response

def main(
    base_model_id="Qwen/Qwen3-0.6B",
    adapter_path_1="models/hyperparameter_sweep/run_2_lr-0.0002_rank-32_epochs-1/final_model",
    adapter_path_2="models/hyperparameter_sweep/run_4_lr-0.0001_rank-64_epochs-1/final_model"
):
    """
    Compares the outputs of the base model and two fine-tuned models.
    """
    complaints = load_complaints()
    
    # --- Load Base Model ---
    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # --- Load Fine-Tuned Model 1 (Run 2) ---
    print("\nLoading fine-tuned model 1 (Run 2: lr=2e-4, rank=32, epochs=1)...")
    ft_model_1 = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ft_model_1 = PeftModel.from_pretrained(ft_model_1, adapter_path_1)
    ft_model_1 = ft_model_1.merge_and_unload()

    # --- Load Fine-Tuned Model 2 (Run 4) ---
    print("\nLoading fine-tuned model 2 (Run 4: lr=1e-4, rank=64, epochs=1)...")
    ft_model_2 = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ft_model_2 = PeftModel.from_pretrained(ft_model_2, adapter_path_2)
    ft_model_2 = ft_model_2.merge_and_unload()

    print("\n--- Starting 3-Way Model Comparison ---\n")

    system_prompt = "You are a helpful customer support agent. Please respond to the following customer complaint with empathy and provide a clear solution."

    for complaint in complaints:
        user_prompt = complaint['vars']['complaint']
        
        print("="*100)
        print(f"Complaint: {user_prompt}")
        print("-"*100)

        # --- Base Model Response ---
        print("BASE MODEL (Qwen3-0.6B Original):")
        base_response = generate_response(base_model, base_tokenizer, system_prompt, user_prompt)
        print(f"   {base_response.strip()}")
        
        # --- Fine-Tuned Model 1 Response ---
        print("\nFINE-TUNED MODEL 1 (Run 2: lr=2e-4, rank=32, epochs=1):")
        ft_response_1 = generate_response(ft_model_1, base_tokenizer, system_prompt, user_prompt)
        print(f"   {ft_response_1.strip()}")

        # --- Fine-Tuned Model 2 Response ---
        print("\nFINE-TUNED MODEL 2 (Run 4: lr=1e-4, rank=64, epochs=1):")
        ft_response_2 = generate_response(ft_model_2, base_tokenizer, system_prompt, user_prompt)
        print(f"   {ft_response_2.strip()}")
        
        print("="*100 + "\n")

if __name__ == "__main__":
    main()
