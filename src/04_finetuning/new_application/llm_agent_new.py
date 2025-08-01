
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BitsAndBytesConfig
)
import torch
from peft import PeftModel

MODEL_CONFIG = {
    "Qwen/Qwen3-0.6B": {
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "type": "causal-chat",
    },
    "Qwen3-0.6B-fine-tuned": {
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "type": "causal-chat",
        "base_model_id": "Qwen/Qwen3-0.6B",
        "adapter_path": "models/Qwen3-0.6B-fine-tuned/final_model",
        "quantized": False, # LoRA model = already optimized
    },
    "google/flan-t5-large": {
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
        "type": "conditional-generation",
    },
    "google/gemma-3-1b-it": {
        "model_class": AutoModelForCausalLM,
        "tokenizer_class": AutoTokenizer,
        "type": "causal-chat",
        "quantized": True,
    },
}

DEFAULT_MODEL = "Qwen3-0.6B-fine-tuned"

class LLMAgent:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = DEFAULT_MODEL
        
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Model '{model_name}' is not supported. Please add it to MODEL_CONFIG.")

        self.model_name = model_name
        self.config = MODEL_CONFIG[self.model_name]
        
        print(f"Loading model: {self.model_name}...")
        
        load_model_id = self.config.get("base_model_id", self.model_name)
        
        self.tokenizer = self.config["tokenizer_class"].from_pretrained(load_model_id)

        model_kwargs = { "device_map": "auto" }

        if self.config.get("quantized"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = "auto"

        self.model = self.config["model_class"].from_pretrained(
            load_model_id,
            **model_kwargs
        )
        
        if "adapter_path" in self.config:
            print(f"Loading adapter from: {self.config['adapter_path']}")
            self.model = PeftModel.from_pretrained(self.model, self.config['adapter_path'])
            self.model = self.model.merge_and_unload()
            print("Adapter loaded and merged successfully.")

        if "gemma" in self.model_name:
            self.model.eval()

        print("Model loaded successfully.")

    def __call__(self, prompt: str, system_prompt: str = None, **kwargs):
        model_type = self.config["type"]

        if model_type == "causal-chat":
            return self._generate_causal_chat(prompt, system_prompt, **kwargs)
        elif model_type == "conditional-generation":
            return self._generate_conditional(prompt, **kwargs)
        else:
            raise NotImplementedError(f"Generation for model type '{model_type}' is not implemented.")

    def _prepare_qwen_input(self, messages, **kwargs):
        enable_thinking = kwargs.pop("enable_thinking", False)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def _prepare_gemma_input(self, messages, **kwargs):
        gemma_messages = [[
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages
        ]]
        return self.tokenizer.apply_chat_template(
            gemma_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

    def _decode_qwen_output(self, output_ids, input_length):
        output_ids = output_ids[0][input_length:].tolist()
        try:
            think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        except (ValueError, IndexError):
            thinking_content = ""
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return content, thinking_content

    def _decode_gemma_output(self, output_ids, input_length):
        content = self.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        return content.strip(), ""
        
    def _generate_causal_chat(self, prompt, system_prompt, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        model_name_for_logic = self.config.get("base_model_id", self.model_name)

        if "Qwen" in model_name_for_logic:
            model_inputs = self._prepare_qwen_input(messages, **kwargs)
        elif "gemma" in model_name_for_logic:
            model_inputs = self._prepare_gemma_input(messages, **kwargs)
        else:
            raise NotImplementedError(f"Causal chat for {model_name_for_logic} is not implemented.")

        default_gen_kwargs = {
            "max_new_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        gen_kwargs = {**default_gen_kwargs, **kwargs}
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )
        
        input_length = model_inputs.input_ids.shape[1]
        if "Qwen" in model_name_for_logic:
            return self._decode_qwen_output(generated_ids, input_length)
        elif "gemma" in model_name_for_logic:
            return self._decode_gemma_output(generated_ids, input_length)
        else:
            output_ids = generated_ids[0][input_length:].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            return content, ""

    def _generate_conditional(self, prompt, **kwargs):
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        default_gen_kwargs = { "max_new_tokens": 512 }
        gen_kwargs = {**default_gen_kwargs, **kwargs}

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        content = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        return content, ""
