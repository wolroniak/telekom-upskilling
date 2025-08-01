import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent.parent / "00_setup"))
sys.path.insert(0, str(current_dir.parent.parent / "02_rag"))
sys.path.insert(0, str(current_dir.parent.parent / "03_agent_decision_logic"))
sys.path.insert(0, str(current_dir.parent.parent / "04_finetuning" / "new_application"))

from llm_agent import LLMAgent
from rag_pipeline import RAGPipeline
import anthropic
import torch
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class SystemEvaluator:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            print("Warning: ANTHROPIC_API_KEY not set. LLM-based evaluation will be skipped.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        
        # customer complaints
        self.test_cases = [
            {
                "complaint": "My internet has been down since yesterday. I work from home and this is very frustrating!",
                "reference": "I'm very sorry to hear that your internet has been down for a full day; I know how critical that is when you're working from home. Let's start by checking for any known outages in your area. Could you please provide your street address or account number so I can investigate that for you right away?"
            },
            {
                "complaint": "I was overcharged on my last bill. Can you help?",
                "reference": "Of course, I can certainly help you with that. I'm sorry for the confusion on your recent bill. To investigate the overcharge, could you please provide your account number and the date of the bill in question? I will review the charges with you and we can work to get this corrected."
            },
            {
                "complaint": "The internet speed is very slow during peak hours",
                "reference": "I understand how frustrating slow internet can be, especially during peak hours. A few common causes for this are network congestion or an issue with the local connection. A good first step is to restart your router. If that doesn't help, we can run a diagnostic on your line to check its performance. Would you like me to start that diagnostic?"
            },
            {
                "complaint": "My service installation was scheduled for today but nobody showed up",
                "reference": "I sincerely apologize that our technician didn't arrive for your scheduled installation today. This is absolutely not the level of service we strive to provide. Let me immediately check what happened and reschedule your installation at the earliest possible time slot. I'll also ensure you receive priority scheduling and a confirmation call. Can I get your order number to expedite this?"
            },
            {
                "complaint": "I can't access my email through your service",
                "reference": "I'm sorry you're having trouble accessing your email. This can be frustrating when you need to stay connected. Let's troubleshoot this step by step. First, can you tell me what error message you're seeing? Also, are you trying to access email through our webmail portal, or are you using an email client like Outlook? This will help me guide you to the right solution."
            }
        ]
        
        # Init BLEU
        self.bleu_metric = BLEU()
        self.results = []

    def evaluate_with_claude(self, response: str) -> Dict[str, int]:
        """Evaluate response using Claude Sonnet 3.5"""
        if not self.client:
            return {"helpfulness": 0, "clarity": 0, "empathy": 0, "safety": 0}
        
        system_prompt = """
        You are an expert evaluator of customer support AI agents. Rate the following response on a scale of 1 to 5 for four metrics:

        - Helpfulness (1-5): Does the response provide concrete, actionable advice to solve the problem?
        - Clarity (1-5): Is the response easy to understand, well-structured, and free of jargon?
        - Empathy (1-5): Does the response acknowledge the customer's frustration with appropriate tone?
        - Safety (1-5): Is the response safe, responsible, and free of harmful content?

        Respond ONLY with a JSON object containing the four scores.
        Example: {"helpfulness": 4, "clarity": 5, "empathy": 5, "safety": 5}
        """
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=150,
                temperature=0,
                system=system_prompt,
                messages=[{
                    "role": "user", 
                    "content": f"Evaluate this customer support response:\n\n{response}"
                }]
            )
            
            scores_text = message.content[0].text
            scores = json.loads(scores_text)
            return scores
            
        except Exception as e:
            print(f"Error evaluating with Claude: {e}")
            return {"helpfulness": 0, "clarity": 0, "empathy": 0, "safety": 0}

    def calculate_bleu_scores(self, response: str, reference: str) -> Dict[str, float]:
        """Calculate BLEU-1, BLEU-2, BLEU-3 scores"""
        try:
            bleu_1 = self.bleu_metric.corpus_score([response], [[reference]]).score
            
            from sacrebleu.metrics import BLEU
            bleu_2_metric = BLEU(max_ngram_order=2)
            bleu_3_metric = BLEU(max_ngram_order=3)
            
            bleu_2 = bleu_2_metric.corpus_score([response], [[reference]]).score
            bleu_3 = bleu_3_metric.corpus_score([response], [[reference]]).score
            
            return {"bleu_1": bleu_1, "bleu_2": bleu_2, "bleu_3": bleu_3}
        except Exception as e:
            print(f"Error calculating BLEU scores: {e}")
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0}

    def calculate_bert_score(self, response: str, reference: str) -> float:
        """Calculate BERTScore for semantic similarity"""
        try:
            _, _, f1_scores = bert_score([response], [reference], lang="en", verbose=False)
            return f1_scores.item()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return 0.0

    def calculate_perplexity(self, response: str, model, tokenizer) -> float:
        """Calculate perplexity for model confidence"""
        try:
            inputs = tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
            
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            return float('inf')

    def calculate_all_metrics(self, response: str, reference: str, model=None, tokenizer=None) -> Dict:
        """Calculate all evaluation metrics for a response"""
        metrics = {}
        
        # LLM-based scores
        claude_scores = self.evaluate_with_claude(response)
        metrics.update(claude_scores)
        
        # BLEU scores
        bleu_scores = self.calculate_bleu_scores(response, reference)
        metrics.update(bleu_scores)
        
        # BERTScore
        metrics["bert_score"] = self.calculate_bert_score(response, reference)
        
        # Perplexity
        if model and tokenizer:
            metrics["perplexity"] = self.calculate_perplexity(response, model, tokenizer)
        else:
            metrics["perplexity"] = None
            
        return metrics

    def run_prompt_only_evaluation(self) -> List[Dict]:
        """Evaluate Prompt-Only system variant"""
        print("### Evaluating Prompt-Only System...")
        
        agent = LLMAgent(model_name="Qwen/Qwen3-0.6B")
        system_prompt = "You are a customer support agent. Respond to the following complaint."
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"  Processing test case {i}/{len(self.test_cases)}")
            
            complaint = test_case["complaint"]
            reference = test_case["reference"]
            
            response, _ = agent(complaint, system_prompt=system_prompt)
            metrics = self.calculate_all_metrics(response, reference, agent.model, agent.tokenizer)
            
            llm_scores = {k: v for k, v in metrics.items() if k in ["helpfulness", "clarity", "empathy", "safety"]}
            average_llm_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0
            
            results.append({
                "test_case": i,
                "complaint": complaint,
                "reference": reference,
                "response": response,
                "metrics": metrics,
                "average_llm_score": average_llm_score
            })
            
        return results

    def run_rag_only_evaluation(self) -> List[Dict]:
        """Evaluate RAG-Only system variant"""
        print("### Evaluating RAG-Only System...")
        
        agent = LLMAgent(model_name="Qwen/Qwen3-0.6B")
        knowledge_base_dir = current_dir.parent.parent / "02_rag" / "knowledge_base"
        rag_pipeline = RAGPipeline(knowledge_base_dir=str(knowledge_base_dir))
        
        results = []
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"  Processing test case {i}/{len(self.test_cases)}")
            
            complaint = test_case["complaint"]
            reference = test_case["reference"]
            
            retrieved_docs, _ = rag_pipeline.retrieve_with_scores(complaint)
            context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            
            rag_prompt = (
                f"Based on the following context, please answer the user's complaint.\n"
                f"Context:\n{context_str}\n\n"
                f"Complaint:\n{complaint}"
            )
            
            response, _ = agent(rag_prompt)
            metrics = self.calculate_all_metrics(response, reference, agent.model, agent.tokenizer)
            
            llm_scores = {k: v for k, v in metrics.items() if k in ["helpfulness", "clarity", "empathy", "safety"]}
            average_llm_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0
            
            results.append({
                "test_case": i,
                "complaint": complaint,
                "reference": reference,
                "response": response,
                "metrics": metrics,
                "average_llm_score": average_llm_score
            })
            
        return results

    def run_finetuned_only_evaluation(self) -> List[Dict]:
        """Evaluate Fine-Tuned Only system variant"""
        print("### Evaluating Fine-Tuned Only System...")
        
        try:
            from llm_agent_new import LLMAgent as LLMAgentNew
            agent = LLMAgentNew(model_name="Qwen3-0.6B-fine-tuned")
            system_prompt = "You are a helpful customer support agent. Please respond to the following customer complaint with empathy and provide a clear solution."
            
            results = []
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"  Processing test case {i}/{len(self.test_cases)}")
                
                complaint = test_case["complaint"]
                reference = test_case["reference"]
                
                response, _ = agent(complaint, system_prompt=system_prompt)
                metrics = self.calculate_all_metrics(response, reference, agent.model, agent.tokenizer)
                
                llm_scores = {k: v for k, v in metrics.items() if k in ["helpfulness", "clarity", "empathy", "safety"]}
                average_llm_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0
                
                results.append({
                    "test_case": i,
                    "complaint": complaint,
                    "reference": reference,
                    "response": response,
                    "metrics": metrics,
                    "average_llm_score": average_llm_score
                })
                
        except Exception as e:
            print(f"  ###  Fine-tuned model not available: {e}")
            print("  ### Using base model as fallback...")
            return self.run_prompt_only_evaluation()
            
        return results

    def run_full_agent_evaluation(self) -> List[Dict]:
        """Evaluate Full Agent (Agent + RAG + Fine-tuning) system variant"""
        print("### Evaluating Full Agent System...")
        
        try:
            from llm_agent_new import LLMAgent as LLMAgentNew
            from decision_agent import DecisionAgent
            
            llm_agent = LLMAgentNew(model_name="Qwen3-0.6B-fine-tuned")
            knowledge_base_dir = current_dir.parent.parent / "02_rag" / "knowledge_base"
            rag_pipeline = RAGPipeline(knowledge_base_dir=str(knowledge_base_dir))
            decision_agent = DecisionAgent(llm_agent, rag_pipeline, threshold=1.0)
            
            results = []
            for i, test_case in enumerate(self.test_cases, 1):
                print(f"  Processing test case {i}/{len(self.test_cases)}")
                
                complaint = test_case["complaint"]
                reference = test_case["reference"]
                
                response, _ = decision_agent.execute_query(complaint)
                metrics = self.calculate_all_metrics(response, reference, llm_agent.model, llm_agent.tokenizer)
                
                llm_scores = {k: v for k, v in metrics.items() if k in ["helpfulness", "clarity", "empathy", "safety"]}
                average_llm_score = sum(llm_scores.values()) / len(llm_scores) if llm_scores else 0
                
                results.append({
                    "test_case": i,
                    "complaint": complaint,
                    "reference": reference,
                    "response": response,
                    "metrics": metrics,
                    "average_llm_score": average_llm_score
                })
                
        except Exception as e:
            print(f"  ###  Full agent not available: {e}")
            print("  ### Using RAG-only as fallback...")
            return self.run_rag_only_evaluation()
            
        return results

    def run_complete_evaluation(self) -> Dict:
        """Run evaluation on all 4 system variants"""
        print("### Starting Comprehensive System Evaluation")
        print("### Metrics: BLEU-1/2/3, BERTScore, Perplexity, LLM-based (Helpfulness, Clarity, Empathy, Safety)")
        print("=" * 80)
        
        start_time = time.time()
        
        evaluation_results = {
            "prompt_only": self.run_prompt_only_evaluation(),
            "rag_only": self.run_rag_only_evaluation(), 
            "finetuned_only": self.run_finetuned_only_evaluation(),
            "full_agent": self.run_full_agent_evaluation()
        }
        
        duration = time.time() - start_time
        
        print("=" * 80)
        print(f"### Evaluation completed in {duration:.1f} seconds")
        
        return evaluation_results

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("# Customer Support LLM Agent Evaluation Report")
        report.append("=" * 60)
        report.append("")
        report.append("## Evaluation Metrics")
        report.append("- **BLEU-1/2/3**: N-gram overlap with reference answers")
        report.append("- **BERTScore**: Semantic similarity to reference answers")
        report.append("- **Perplexity**: Model confidence (lower is better)")
        report.append("- **LLM-based**: Claude Sonnet 3.5 scoring for Helpfulness, Clarity, Empathy, Safety")
        report.append("")
        
        # Summary table
        report.append("## System Variant Comparison")
        report.append("")
        report.append("| System | BLEU-1 | BLEU-2 | BLEU-3 | BERTScore | Perplexity | Help | Clarity | Empathy | Safety | LLM Avg |")
        report.append("|--------|--------|--------|--------|-----------|------------|------|---------|---------|---------|---------|")
        
        for variant_name, variant_results in results.items():
            if not variant_results:
                continue
                
            # averages
            avg_bleu1 = sum(r["metrics"]["bleu_1"] for r in variant_results) / len(variant_results)
            avg_bleu2 = sum(r["metrics"]["bleu_2"] for r in variant_results) / len(variant_results)
            avg_bleu3 = sum(r["metrics"]["bleu_3"] for r in variant_results) / len(variant_results)
            avg_bert = sum(r["metrics"]["bert_score"] for r in variant_results) / len(variant_results)
            
            perplexities = [r["metrics"]["perplexity"] for r in variant_results if r["metrics"]["perplexity"] is not None]
            avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float('inf')
            
            avg_helpfulness = sum(r["metrics"]["helpfulness"] for r in variant_results) / len(variant_results)
            avg_clarity = sum(r["metrics"]["clarity"] for r in variant_results) / len(variant_results)
            avg_empathy = sum(r["metrics"]["empathy"] for r in variant_results) / len(variant_results)
            avg_safety = sum(r["metrics"]["safety"] for r in variant_results) / len(variant_results)
            avg_llm = sum(r["average_llm_score"] for r in variant_results) / len(variant_results)
            
            perp_str = f"{avg_perplexity:.1f}" if avg_perplexity != float('inf') else "N/A"
            
            report.append(f"| {variant_name.replace('_', ' ').title()} | {avg_bleu1:.3f} | {avg_bleu2:.3f} | {avg_bleu3:.3f} | {avg_bert:.3f} | {perp_str} | {avg_helpfulness:.2f} | {avg_clarity:.2f} | {avg_empathy:.2f} | {avg_safety:.2f} | {avg_llm:.2f} |")
        
        report.append("")
        
        for variant_name, variant_results in results.items():
            if not variant_results:
                continue
                
            report.append(f"## {variant_name.replace('_', ' ').title()} - Detailed Results")
            report.append("")
            
            for result in variant_results:
                report.append(f"### Test Case {result['test_case']}")
                report.append(f"**Complaint:** {result['complaint']}")
                report.append(f"**Response:** {result['response']}")
                report.append("")
                report.append("**Metrics:**")
                for metric, value in result['metrics'].items():
                    if value is not None:
                        if isinstance(value, float):
                            report.append(f"- {metric}: {value:.3f}")
                        else:
                            report.append(f"- {metric}: {value}")
                report.append("")
        
        return "\n".join(report)

    def save_results(self, results: Dict, report: str):
        """Save evaluation results and report"""
        output_dir = current_dir
        
        # JSON results
        with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # markdown report
        with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"### Results saved to {output_dir / 'evaluation_results.json'}")
        print(f"### Report saved to {output_dir / 'evaluation_report.md'}")


def main():
    """Main evaluation function"""
    evaluator = SystemEvaluator()
    results = evaluator.run_complete_evaluation()
    report = evaluator.generate_report(results)
    evaluator.save_results(results, report)
    
    print("\n" + "=" * 80)
    print("### EVALUATION SUMMARY")
    print("=" * 80)
    
    summary_start = report.find("## System Variant Comparison")
    summary_end = report.find("## ", summary_start + 1)
    if summary_start != -1:
        summary_section = report[summary_start:summary_end if summary_end != -1 else None]
        print(summary_section)


if __name__ == "__main__":
    main()