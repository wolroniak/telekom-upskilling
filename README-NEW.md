# Telekom Customer Support LLM Agent - Project Guide

**A comprehensive implementation of an intelligent customer support system combining prompt engineering, retrieval-augmented generation, agent decision logic, and fine-tuning.**

---

## Project Overview

This project implements a complete customer support AI assistant that intelligently processes customer complaints and provides contextually appropriate responses. The system decides whether to use retrieved knowledge from a technical database or rely on the model's internal understanding based on query relevance.

### Core Components
- **Base LLM**: Qwen/Qwen3-0.6B with multiple model support
- **Knowledge Base**: Technical documentation covering billing, troubleshooting, and policies  
- **Decision Agent**: Intelligent routing between RAG and direct LLM responses
- **Fine-tuning**: LoRA-based adaptation for improved customer service tone
- **Evaluation**: Multi-metric assessment framework

---

## Quick Start Guide

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage Examples

#### 1. Test the Base Agent
```bash
cd src/00_setup
python run_agent.py
```
This demonstrates the core LLM agent with different models (Qwen, Gemma, T5).

#### 2. Try Different Prompt Templates
```bash
cd src/01_prompt_engineering
python run_prompts.py
```
See how empathetic, structured, and friendly prompts perform on customer complaints.

#### 3. Test RAG System
```bash
cd src/02_rag
python run_rag_evaluation.py
```
Experience how the system retrieves relevant context from the knowledge base.

#### 4. Run the Decision Agent
```bash
cd src/03_agent_decision_logic
python run_decision_agent.py
```
Watch the agent decide between RAG and LLM-only responses based on relevance scores.

#### 5. Test Fine-tuned Model (after training)
```bash
cd src/04_finetuning/new_application
python run_final_agent.py
```
Use the complete system with fine-tuned model, RAG, and decision logic.

---

## System Architecture

### Phase 1: Prompt Engineering (`src/01_prompt_engineering/`)
**Purpose**: Design optimal prompts for customer support scenarios

- **Input**: Customer complaints from `complaints.json`
- **Templates**: Three variants (empathetic, structured, friendly) in `prompts/`
- **Evaluation**: Automated testing with Promptfoo framework
- **Output**: Performance comparison across prompt types and models

**Key File**: `run_prompts.py` - Execute all prompt variants

### Phase 2: RAG Implementation (`src/02_rag/`)
**Purpose**: Augment responses with relevant technical knowledge

- **Knowledge Base**: 6 documents in `knowledge_base/` covering:
  - Billing FAQs and policies
  - Internet troubleshooting procedures  
  - Customer service best practices
- **Retrieval**: FAISS vector store with sentence transformers
- **Pipeline**: `rag_pipeline.py` handles document loading, embedding, and retrieval

**Key File**: `rag_pipeline.py` - Core retrieval functionality

### Phase 3: Agent Decision Logic (`src/03_agent_decision_logic/`)
**Purpose**: Intelligent routing between RAG and LLM-only responses

- **Decision Function**: Score-based threshold system
- **Threshold**: Optimized value (1.0) determines routing
- **Logging**: Complete decision process and reasoning
- **Integration**: Combines LLM agent with RAG pipeline

**Key File**: `decision_agent.py` - Core decision logic

### Phase 4: Fine-tuning (`src/04_finetuning/`)
**Purpose**: Adapt model for improved customer service tone and helpfulness

- **Method**: LoRA (Low-Rank Adaptation) with 4-bit quantization
- **Datasets**: Alpaca (instruction-following) + Anthropic HH (helpfulness/safety)
- **Training**: `fine_tuning.py` with configurable hyperparameters
- **Optimization**: Systematic hyperparameter sweep (`run_hyperparameter_sweep.py`)

**Key Files**: 
- `data_preparation.py` - Dataset processing
- `fine_tuning.py` - Training pipeline
- `evaluation.py` - Model comparison

### Phase 5: Evaluation (`src/05_evaluation/`)
**Purpose**: Comprehensive assessment across all system variants

- **Metrics**: BLEU, BERTScore, Perplexity, LLM-based feedback
- **Variants**: Prompt-only, RAG-only, Fine-tuned-only, Full Agent
- **Framework**: Custom evaluation system with Claude Sonnet 3.5 assessment
- **Output**: Detailed performance reports and comparisons

**Key File**: `custom_evaluation/evaluator.py` - Complete evaluation suite

---

## Understanding the Data Flow

### 1. Customer Complaint Processing
```
Customer Complaint → Agent Decision Logic → Route Selection
                                        ↓
                              RAG Path ←→ LLM-Only Path
                                        ↓
                              Generated Response → Evaluation
```

### 2. RAG Decision Process
```python
# Example from decision_agent.py
if retrieval_score >= threshold:
    # Use RAG path - high relevance found
    context = rag_pipeline.retrieve(complaint)
    response = llm_agent(complaint + context)
else:
    # Use LLM-only path - low relevance
    response = llm_agent(complaint)
```

### 3. Knowledge Base Structure
```
src/02_rag/knowledge_base/
├── billing_faq.txt                    # Payment and billing issues
├── internet_troubleshooting.txt       # Basic connectivity fixes
├── service_policies.txt               # SLA and policy info
├── website_customer_service_best_practices.txt
├── website_internet_troubleshooting.txt
└── windows_fix_wifi_connection_in_windows.txt
```

---

## Model Configurations

### Supported Models
```python
# From src/00_setup/llm_agent.py
MODEL_CONFIG = {
    "Qwen/Qwen3-0.6B": {"type": "causal", "thinking": True},
    "google/flan-t5-large": {"type": "seq2seq", "thinking": False},
    "google/gemma-3-1b-it": {"type": "causal", "thinking": False}
}
```

### Fine-tuning Configuration
```python
# Example hyperparameters
{
    "learning_rate": 2e-4,
    "lora_rank": 32,
    "lora_alpha": 16,
    "num_train_epochs": 1,
    "use_4bit": True
}
```

---

## Evaluation Metrics Explained

### Automated Metrics
- **BLEU-1/2/3**: Measures word overlap with reference answers
- **BERTScore**: Semantic similarity using BERT embeddings  
- **Perplexity**: Model confidence (lower = more confident)

### LLM-based Assessment (Claude Sonnet 3.5)
- **Helpfulness** (1-5): How well the response addresses the customer's need
- **Clarity** (1-5): How clear and understandable the response is
- **Empathy** (1-5): How well the response acknowledges customer emotions
- **Safety** (1-5): Whether the response avoids harmful or inappropriate content

---

## Key Results Summary

### System Performance
| System | Strength | Best Use Case |
|--------|----------|---------------|
| **Prompt-Only** | Clarity (4.4/5) | Simple acknowledgments |
| **RAG-Only** | Helpfulness (4.0/5) | Technical troubleshooting |
| **Fine-tuned** | Balanced tone | Customer empathy |
| **Full Agent** | Empathy (4.4/5) | Complete customer support |

### Decision Logic Performance
- **Technical queries → RAG**: 60% (internet, billing, speed issues)
- **Service queries → LLM**: 40% (installation, general inquiries)
- **Accuracy**: 100% correct routing decisions

---

## Running Your Own Experiments

### Modify Knowledge Base
1. Add new documents to `src/02_rag/knowledge_base/`
2. Restart RAG pipeline to rebuild vector store
3. Test with relevant customer complaints

### Adjust Decision Threshold
```python
# In src/03_agent_decision_logic/run_decision_agent.py
decision_agent = DecisionAgent(llm_agent, rag_pipeline, threshold=0.8)
# Lower threshold = more RAG usage
# Higher threshold = more LLM-only usage
```

### Add New Test Cases
1. Edit `src/05_evaluation/custom_evaluation/evaluator.py`
2. Add complaints and reference answers to `test_cases`
3. Run evaluation: `python src/05_evaluation/run_custom_evaluation.py`

### Fine-tune with Different Data
1. Modify `src/04_finetuning/data_preparation.py`
2. Adjust dataset sources or formatting
3. Run training: `python src/04_finetuning/fine_tuning.py`

---

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all requirements installed with `pip install -r requirements.txt`
2. **Memory Issues**: Use 4-bit quantization for large models
3. **Slow Performance**: Reduce batch size or use smaller models for testing
4. **CUDA Errors**: Set device to "cpu" if GPU unavailable

### Debugging Tools
- **Logging**: All components include detailed logging
- **Evaluation**: Test individual components before full integration
- **TensorBoard**: Monitor training progress in `models/hyperparameter_sweep/`

---

## File Structure Overview
```
telekom-upskilling/
├── src/
│   ├── 00_setup/           # Core LLM agent
│   ├── 01_prompt_engineering/  # Prompt templates and evaluation
│   ├── 02_rag/             # Knowledge base and retrieval
│   ├── 03_agent_decision_logic/  # Decision routing
│   ├── 04_finetuning/      # Model adaptation
│   └── 05_evaluation/      # Performance assessment
├── prompts/                # Template definitions
├── models/                 # Trained model storage
├── results/                # Evaluation outputs
└── requirements.txt        # Python dependencies
```

This project demonstrates a complete AI customer support pipeline from basic prompting through advanced fine-tuning and evaluation. Each phase builds upon the previous to create a sophisticated, production-ready system.