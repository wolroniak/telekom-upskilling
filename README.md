# Customer Support LLM Agent - Project

---

## Project Overview

This project implements a customer support AI assistant that intelligently processes customer complaints and provides contextually appropriate responses. The system decides whether to use retrieved knowledge from a knowledge base or rely on the model's internal understanding based on query relevance.

### Core Components
- **Base LLM**: Qwen/Qwen3-0.6B with multiple other model support
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
This demonstrates how empathetic, structured, and friendly prompts perform on customer complaints.

#### 3. Test RAG System
```bash
cd src/02_rag
python run_rag_evaluation.py
```
This demonstrates how the system retrieves relevant context from the knowledge base.

#### 4. Run the Decision Agent
```bash
cd src/03_agent_decision_logic
python run_decision_agent.py
```
This demonstrates how the agent decides between RAG and LLM-only responses based on relevance scores.

#### 5. Test Fine-tuned Model (after training)
```bash
cd src/04_finetuning/new_application
python run_final_agent.py
```
This demonstrates how to use the complete system with fine-tuned model, RAG, and decision logic.

---

### Knowledge Base Structure
```
src/02_rag/knowledge_base/
├── billing_faq.txt                    # synthetic
├── internet_troubleshooting.txt       # synthetic
├── service_policies.txt               # synthetic
├── website_customer_service_best_practices.txt
├── website_internet_troubleshooting.txt
└── windows_fix_wifi_connection_in_windows.txt
```

---

## Key Results Summary

The full evaluation report can be found here: **src\05_evaluation\custom_evaluation\evaluation_report.md**

---

## File Structure Overview
```
telekom-upskilling/
├── src/
│   ├── 00_setup/                   # Core LLM agent
│   ├── 01_prompt_engineering/      # Prompt templates and evaluation
│   ├── 02_rag/                     # Knowledge base and retrieval
│   ├── 03_agent_decision_logic/    # Decision routing
│   ├── 04_finetuning/              # Model adaptation
│   └── 05_evaluation/              # Performance assessment
├── prompts/                        # Template definitions
├── models/                         # Trained model storage
├── results/                        # Evaluation outputs
└── requirements.txt                # Python dependencies
```

This project is one solution for the telekom-upskilling-certification and demonstrates a LLM customer support pipeline from basic prompting through fine-tuning and evaluation. Each phase builds upon the previous to see the evolution of the project.