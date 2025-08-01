# telekom-upskilling

Building and Evaluating a Helpful, Retrieval-Augmented Customer Support LLM Agent.

## Objective: 
The goal of this project is to design, implement, and evaluate a lightweight, locally deployable customer support assistant that generates consistently helpful, accurate, and safe responses to user complaints. The assistant will integrate: 
• Prompt engineering for controllable behaviour, 
• Retrieval-Augmented Generation (RAG) for factual grounding, 
• Agent-based decision logic to choose between internal or external knowledge, 
• Fine-tuning using instruction-following and safety-aligned datasets. 

A key focus is maximizing the usefulness, politeness, and trustworthiness of outputs, while minimizing the risk of unhelpful, rude, or factually incorrect responses. Evaluation will use automated metrics and LLM-based feedback to ensure improvements are measurable and aligned with end-user expectations. 

### Base Model Selection 
Throughout all stages, use the same compact base model to ensure fair and consistent 
evaluation. Suggested options: 
- Flan-T5-Small (~80M parameters)  
- Phi-2 (~1.3B)  
- Mistral 7B-Instruct 

## Project Steps 
### 1. Prompt Engineering for Customer Support
Dataset: 
- Create a small synthetic dataset of 5 - 10 common customer complaints (e.g., internet outage, billing issue, slow speed). 
Example complaints: 
	- "My internet has been down since yesterday. I work from home and this is very frustrating!" 
	- "I was overcharged on my last bill. Can you help?" 
Task: 
	- Design 3 prompt templates (empathetic, structured, friendly) for these complaints. 
	- Use Promptfoo to evaluate prompts on your dataset for empathy, clarity, and usefulness. 
	- Record and compare prompt effectiveness across templates and model variants. 

### 2. Retrieval-Augmented Generation (RAG)
Dataset:
- Create a small knowledge base using: 
	- 2–3 curated Wikipedia articles (e.g., Troubleshooting Internet Connection, Customer Service Best Practices) 
	- Company FAQ files (sample or synthetic) 
Task: 
- Build a minimal retrieval pipeline using sentence-transformers or similar. 
- For each complaint, retrieve relevant context and combine it with the user query in the prompt to the LLM. 
- Evaluate if the retrieved chunks are relevant and improve the answer quality over prompt-only responses. 

### 3. Agent Decision Logic
Dataset: 
- Use the same set of customer complaints and knowledge base as above.
Task:   
- Implement a simple decision function: 
	- If the retrieval score is high, use RAG; otherwise, answer directly using the LLM’s internal knowledge. 
- Log the agent’s decision and the final response for each query. 

### 4. Fine-Tuning for Helpfulness and Safety - 
Dataset: 
- Use a subset (2,000–3,000 samples) of the Alpaca dataset for instruction following. 
- Anthropic Helpful-Harmless (HH) dataset: 
	- Focus on examples that contrast helpful vs. evasive, or harmless vs. harmful completions. 
	- Optionally, rephrase examples to resemble customer service tone and contexts. 
- Optional: Custom examples of good and bad responses (e.g., rude tone, hallucinations, polite apology).
Task: 
- Fine-tune a compact open-source LLM (e.g., Flan-T5-Small,  or Mistral 7B Instruct) using LoRA and Hugging Face PEFT. 
- Compare model outputs before and after fine-tuning on your test set of customer complaints. 
Subtask (Hyperparameter Selection): 
- Explore and compare training configurations: 
	- Learning rate (e.g., 1e-5, 5e-5, 1e-4) 
	- Epochs, batch size, gradient accumulation 
	- LoRA rank and dropout 
- Track validation loss, fluency, and helpfulness scores to choose the best setup. 

### 5. Evaluation 
Datasets: 
- Use your synthetic complaints dataset 
- Optional: CNN/DailyMail for broader generalization checks 
Metrics: 
- BLEU-1/2/3 for n-gram overlap (if you have reference answers) 
- BERTScore for semantic similarity  
- Perplexity for model confidence 
- LLM-based feedback: 
	- Use GPT-4 or a small local reward model to evaluate responses based on: 
		- Helpfulness 
		- Clarity 
		- Empathy 
		- Safety 
Task: 
- Compare variants of the system: 
	- Prompt-only 
	- RAG-only 
- Fine-tuned-only 
- Agent + RAG + Fine-tuning 


## Reporting - Important for getting the certification 
Deliverables: 
- Code/scripts for each component (prompt templates, retrieval, agent logic, fine tuning, evaluation) 
- Example queries, agent decisions, and generated responses 
- Evaluation results (metrics and LLM-based feedback) 
- Final short report (1–2 pages) covering: 
	- Hyperparameter tuning report: 
	- What worked well 
	- Key challenges
