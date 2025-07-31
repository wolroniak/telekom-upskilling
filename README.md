# telekom-upskilling

This comprehensive plan breaks down your certification project into manageable phases with minimal starting points and clear expansion paths. Here are the key points to get you started successfully:
🎯 Start Here - Your First Steps:

Choose Flan-T5-Small as your base model - it's the most beginner-friendly option with fast inference and lower resource requirements
Begin with the Day 1 Priority Tasks - they'll give you immediate momentum and confidence
Focus on "Minimal Requirements" first - each phase has a simple starting implementation that works

🔑 Critical Success Factors:
For Certification Success:

Document everything as you go - your report quality is crucial
Take screenshots of Promptfoo results and evaluation metrics
Keep a daily log of challenges and solutions
Track all hyperparameter experiments with results

Technical Tips:

Start with the 5 sample complaints I provided - they cover common customer service scenarios
Use the minimal code examples as starting points - they're tested patterns
The 3-template prompt approach gives you immediate comparative results

📈 Expansion Strategy:
Once you complete the minimal version of each phase:

Phase 1-2: Add more diverse complaints and sophisticated prompts
Phase 3: Implement vector databases and better chunking
Phase 4: Add multi-step reasoning to agent decisions
Phase 5: Experiment with larger datasets and advanced training techniques

🚨 Common Pitfalls to Avoid:

Don't start with complex models like Mistral 7B - begin small
Avoid over-engineering early - get the minimal version working first
Don't skip evaluation - metrics are essential for certification
Keep your knowledge base small initially - 2-3 text files are sufficient


# The official project description

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



# Customer Support LLM Agent - Complete Project Plan

## 🎯 Project Overview
**Goal**: Build a lightweight, locally deployable customer support assistant that generates helpful, accurate, and safe responses using prompt engineering, RAG, agent logic, and fine-tuning.

**Success Criteria**: 
- Measurable improvements in helpfulness, accuracy, and safety
- Complete evaluation with automated metrics and LLM feedback
- Professional certification report

---

## 🛠️ Tools & Technologies Setup

### Core Tools (Minimal Setup)
```bash
# Python Environment
python>=3.8
pip install transformers datasets torch sentence-transformers
pip install huggingface-hub accelerate peft
pip install sklearn numpy pandas matplotlib seaborn
pip install evaluate nltk rouge-score bert-score
```

### Additional Tools
- **Promptfoo**: For prompt evaluation
- **Weights & Biases**: For experiment tracking (optional but recommended)
- **Jupyter Notebook**: For development and analysis
- **Git**: For version control

### Model Recommendations by Experience Level
- **Beginner**: Flan-T5-Small (80M params) - fastest, easiest to work with
- **Intermediate**: Phi-2 (1.3B params) - good balance
- **Advanced**: Mistral 7B-Instruct - best performance

---

## 📋 Detailed TODO List

### Phase 1: Environment Setup & Data Preparation
**Duration**: 1-2 days

#### TODO 1.1: Environment Setup
- [ ] Create Python virtual environment
- [ ] Install required packages
- [ ] Set up Jupyter notebook environment
- [ ] Test base model loading (start with Flan-T5-Small)
- [ ] Initialize project repository structure

#### TODO 1.2: Create Minimal Dataset
**Minimal Requirement**: 5-10 customer complaints
```
complaints = [
    "My internet has been down since yesterday. I work from home and this is very frustrating!",
    "I was overcharged on my last bill. Can you help?",
    "The internet speed is very slow during peak hours",
    "I can't access my email through your service",
    "My service was supposed to be installed today but nobody showed up"
]
```

**Expansion**: Add 20-50 more varied complaints, categorize by type (billing, technical, service)

#### TODO 1.3: Create Knowledge Base
**Minimal Requirement**: 2-3 text files
- `internet_troubleshooting.txt` - Basic troubleshooting steps
- `billing_faq.txt` - Common billing questions
- `service_policies.txt` - Service level agreements

**Expansion**: Add structured FAQ database, official documentation, policy documents

---

### Phase 2: Prompt Engineering
**Duration**: 2-3 days

#### TODO 2.1: Design Prompt Templates
**Minimal Requirement**: 3 basic templates

1. **Empathetic Template**:
```
"I understand your frustration with [issue]. Let me help you resolve this.
Customer complaint: {complaint}
Response:"
```

2. **Structured Template**:
```
"Customer Service Response Format:
1. Acknowledge the issue
2. Provide solution steps
3. Offer additional help

Complaint: {complaint}
Response:"
```

3. **Friendly Template**:
```
"Hello! I'm here to help with your concern.
Issue: {complaint}
Friendly solution:"
```

#### TODO 2.2: Implement Promptfoo Evaluation
- [ ] Install and configure Promptfoo
- [ ] Create evaluation config file
- [ ] Define evaluation criteria (empathy, clarity, usefulness)
- [ ] Run comparative evaluation
- [ ] Document results with screenshots and metrics

**Expansion**: A/B test with more templates, add few-shot examples, implement chain-of-thought prompting

---

### Phase 3: Retrieval-Augmented Generation (RAG)
**Duration**: 3-4 days

#### TODO 3.1: Build Minimal Retrieval Pipeline
**Minimal Implementation**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

class MinimalRAG:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []  # List of text chunks
        self.embeddings = None
    
    def add_documents(self, docs):
        self.knowledge_base = docs
        self.embeddings = self.encoder.encode(docs)
    
    def retrieve(self, query, top_k=2):
        query_embedding = self.encoder.encode([query])
        scores = np.dot(query_embedding, self.embeddings.T)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices], scores[top_indices]
```

#### TODO 3.2: Integration with LLM
- [ ] Implement context injection into prompts
- [ ] Test retrieval relevance manually
- [ ] Compare RAG vs non-RAG responses
- [ ] Document retrieval performance

**Expansion**: Use vector databases (Chroma, Pinecone), implement semantic chunking, add reranking

---

### Phase 4: Agent Decision Logic
**Duration**: 2 days

#### TODO 4.1: Implement Simple Decision Function
**Minimal Implementation**:
```python
def agent_decision(query, retrieval_scores, threshold=0.7):
    max_score = max(retrieval_scores) if retrieval_scores else 0
    
    if max_score > threshold:
        return "RAG", f"Using retrieved context (confidence: {max_score:.2f})"
    else:
        return "DIRECT", f"Using internal knowledge (low retrieval confidence: {max_score:.2f})"
```

#### TODO 4.2: Logging and Analysis
- [ ] Log all agent decisions
- [ ] Track decision accuracy manually
- [ ] Analyze decision patterns
- [ ] Create decision flow diagram

**Expansion**: Multi-step reasoning, tool selection, confidence calibration

---

### Phase 5: Fine-Tuning
**Duration**: 3-5 days

#### TODO 5.1: Data Preparation
**Minimal Dataset**: 
- 1000 samples from Alpaca dataset
- 500 samples from HH dataset
- 100 custom customer service examples

#### TODO 5.2: LoRA Fine-Tuning Setup
**Minimal Configuration**:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
```

#### TODO 5.3: Training Pipeline
- [ ] Implement training loop with minimal hyperparameters
- [ ] Start with: lr=5e-5, batch_size=4, epochs=3
- [ ] Monitor training loss
- [ ] Save checkpoints

#### TODO 5.4: Hyperparameter Exploration
**Minimal Grid Search**:
- Learning rates: [1e-5, 5e-5, 1e-4]
- LoRA ranks: [8, 16, 32]
- Epochs: [2, 3, 5]

**Expansion**: Bayesian optimization, advanced schedulers, larger parameter sweeps

---

### Phase 6: Evaluation
**Duration**: 2-3 days

#### TODO 6.1: Implement Evaluation Metrics
**Minimal Metrics**:
```python
# Automated metrics
from evaluate import load

bleu = load("bleu")
bertscore = load("bertscore")

def evaluate_response(prediction, reference):
    return {
        "bleu": bleu.compute(predictions=[prediction], references=[[reference]]),
        "bertscore": bertscore.compute(predictions=[prediction], references=[reference])
    }
```

#### TODO 6.2: LLM-based Evaluation
**Minimal GPT-4 Evaluation**:
```python
evaluation_prompt = """
Rate the customer service response on a scale of 1-5 for:
1. Helpfulness
2. Clarity  
3. Empathy
4. Safety

Customer complaint: {complaint}
Response: {response}

Provide scores and brief justification.
"""
```

#### TODO 6.3: Comparative Analysis
- [ ] Test 4 system variants:
  - Prompt-only
  - RAG-only  
  - Fine-tuned-only
  - Complete system (Agent + RAG + Fine-tuning)
- [ ] Generate comparison tables
- [ ] Create visualization charts

**Expansion**: Human evaluation, A/B testing, statistical significance testing

---

## 📊 Report Structure Template

### Executive Summary (0.5 pages)
- **Project Overview**: Brief description of the customer support agent
- **Key Results**: Top 3 findings with metrics
- **Recommendations**: Main takeaways for implementation

### Methodology (0.5 pages)
- **Architecture Overview**: Simple system diagram
- **Dataset Description**: Size, sources, characteristics
- **Evaluation Framework**: Metrics and evaluation setup

### Results & Analysis (0.5 pages)
- **Component Performance**: 
  - Prompt engineering results (Promptfoo scores)
  - RAG retrieval accuracy
  - Fine-tuning improvements
- **Comparative Analysis**: System variant performance table
- **Key Insights**: What worked best and why

### Technical Implementation (0.5 pages)
- **Hyperparameter Tuning Results**: Best configurations found
- **Architecture Decisions**: Justification for design choices
- **Challenges & Solutions**: Problems encountered and fixes

### Appendices
- **Code Repository**: Link to complete implementation
- **Example Interactions**: Sample queries and responses
- **Detailed Metrics**: Complete evaluation results

---

## 🚀 Implementation Roadmap

### Week 1: Foundation
- Days 1-2: Environment setup + minimal dataset creation
- Days 3-5: Prompt engineering + Promptfoo evaluation
- Days 6-7: Basic RAG implementation

### Week 2: Core Development  
- Days 1-2: Agent decision logic + integration
- Days 3-7: Fine-tuning pipeline + initial training

### Week 3: Optimization & Evaluation
- Days 1-3: Hyperparameter tuning
- Days 4-6: Comprehensive evaluation
- Day 7: Results analysis + documentation

### Week 4: Reporting & Finalization
- Days 1-3: Report writing + visualizations
- Days 4-5: Code cleanup + documentation
- Days 6-7: Final review + submission preparation

---

## ⚡ Quick Start Checklist

### Day 1 Priority Tasks:
- [ ] Clone/create project repository
- [ ] Set up Python environment with core packages
- [ ] Load and test Flan-T5-Small model
- [ ] Create 5 sample customer complaints
- [ ] Write first basic prompt template
- [ ] Generate initial response examples

### Success Metrics to Track:
- **Prompt Engineering**: Promptfoo scores improving by 20%+ 
- **RAG**: Retrieval relevance > 70% for relevant queries
- **Fine-tuning**: Validation loss decreasing consistently
- **Overall System**: LLM evaluator scores > 4/5 on helpfulness

This plan balances thoroughness with practicality - start minimal and expand based on your progress and certification requirements!