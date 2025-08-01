
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
    
    # Use the chat template to format the input, ensuring thinking is disabled
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False 
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, eos_token_id=tokenizer.eos_token_id)
    
    # Decode the full output and then remove the prompt part
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The response starts after the 'assistant' token.
    parts = full_response.split('assistant')
    clean_response = parts[-1].strip() if len(parts) > 1 else full_response
    # Remove the empty think tags and any residual newlines
    clean_response = clean_response.replace("<think>", "").replace("</think>", "").strip()
    return clean_response

def main(
    base_model_id="Qwen/Qwen3-0.6B",
    adapter_path="models/Qwen3-0.6B-fine-tuned/final_model"
):
    """
    Compares the outputs of the base model and the fine-tuned model.
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

    # --- Load Fine-Tuned Model ---
    print("\nLoading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, adapter_path)
    ft_model = ft_model.merge_and_unload() # Merge adapter for faster inference

    print("\n--- Starting Evaluation ---\n")

    system_prompt = "You are a helpful customer support agent. Please respond to the following customer complaint with empathy and provide a clear solution."

    for complaint in complaints:
        user_prompt = complaint['vars']['complaint']
        
        print("="*80)
        print(f"Complaint: {user_prompt}")
        print("-"*80)

        # --- Base Model Response ---
        print("Generating response from BASE model...")
        base_response = generate_response(base_model, base_tokenizer, system_prompt, user_prompt)
        print(f"Response (Base): {base_response.strip()}")
        
        # --- Fine-Tuned Model Response ---
        print("\nGenerating response from FINE-TUNED model...")
        ft_response = generate_response(ft_model, base_tokenizer, system_prompt, user_prompt)
        print(f"Response (Fine-Tuned): {ft_response.strip()}")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
