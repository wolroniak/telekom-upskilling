import yaml
import subprocess
import os
import json
import sys
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    spec = importlib.util.spec_from_file_location(
        "rag_pipeline", 
        os.path.join(os.path.dirname(__file__), 'rag_pipeline.py')
    )
    rag_pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_pipeline_module)
    RAGPipeline = rag_pipeline_module.RAGPipeline
except (FileNotFoundError, AttributeError) as e:
    print(f"Error importing RAGPipeline: {e}")
    print("Please ensure 'rag_pipeline.py' exists in the same directory.")
    sys.exit(1)

# --- Configuration ---
EVALUATION_CONFIG = {
    "models": [
        "Qwen/Qwen3-0.6B",
        "google/gemma-3-1b-it",
    ],
    "prompt_files": [
        "prompts/rag_empathetic_prompt.txt",
        "prompts/rag_friendly_prompt.txt",
        "prompts/rag_structured_prompt.txt"
    ],
    "complaints_file": "src/01_prompt_engineering/complaints.json",
    "grading_provider": "anthropic:messages:claude-3-5-sonnet-20241022",
    "results_dir": "results",
}

def generate_rag_test_cases(rag_pipeline, complaints_file):
    """Generates test cases by pairing complaints with retrieved context."""
    print("--- Generating RAG test cases ---")
    try:
        with open(complaints_file, 'r', encoding='utf-8') as f:
            complaints_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Complaints file not found at {complaints_file}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {complaints_file}")
        return None

    test_cases = []
    for item in complaints_data:
        complaint = item.get("vars", {}).get("complaint")
        if not complaint:
            continue

        print(f"Retrieving context for: \"{complaint[:50]}...\"")
        context = rag_pipeline.retrieve(complaint)
        
        test_cases.append({
            "vars": {
                "complaint": complaint,
                "context": context
            }
        })
    
    temp_tests_file = "temp_rag_tests.json"
    with open(temp_tests_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Generated {len(test_cases)} test cases and saved to {temp_tests_file}")
    return temp_tests_file

def run_rag_evaluation():
    """Orchestrates the full RAG evaluation workflow."""
    print("--- Initializing RAG Pipeline ---")
    rag_pipeline = RAGPipeline()
    
    test_cases_file = generate_rag_test_cases(rag_pipeline, EVALUATION_CONFIG["complaints_file"])
    if not test_cases_file:
        return

    os.makedirs(EVALUATION_CONFIG["results_dir"], exist_ok=True)

    for model_id in EVALUATION_CONFIG["models"]:
        for prompt_file in EVALUATION_CONFIG["prompt_files"]:
            prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
            model_name = model_id.split('/')[-1]
            output_file = os.path.join(
                EVALUATION_CONFIG["results_dir"], 
                f"results_{prompt_name}_{model_name}.json"
            )
            
            print(f"\n--- Running evaluation for model: {model_name} with prompt: {prompt_name} ---")

            config = {
                "prompts": [prompt_file],
                "providers": [{
                    "id": "python:prompts/promptfoo_provider.py:call_agent",
                    "config": {"model": model_id}
                }],
                "tests": test_cases_file,
                "defaultTest": {
                    "assert": [
                        {
                            "type": "llm-rubric",
                            "value": "Is the answer helpful and directly based on the internal knowledge, without mentioning that it has internal knowledge?",
                            "provider": EVALUATION_CONFIG["grading_provider"]
                        },
                        {
                            "type": "llm-rubric",
                            "value": "Does the answer match the persona (empathetic, friendly, or structured) defined in the prompt?",
                             "provider": EVALUATION_CONFIG["grading_provider"]
                        }
                    ]
                },
                "outputPath": output_file
            }
            
            temp_config_file = f"temp_rag_{prompt_name}_{model_name}_config.yaml"
            with open(temp_config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
            command = [
                "npx", "promptfoo", "eval",
                "-c", temp_config_file,
                "--max-concurrency", "1"
            ]

            try:
                subprocess.run(command, check=True, shell=True)
                print(f"--- Evaluation complete. Results saved to {output_file} ---")
            except subprocess.CalledProcessError as e:
                print(f"Error running promptfoo for {model_id} with {prompt_file}: {e}")
            except FileNotFoundError:
                print("Error: 'npx' command not found. Please ensure Node.js and npm are installed and in your PATH.")
            finally:
                os.remove(temp_config_file)
                
    os.remove(test_cases_file)

if __name__ == "__main__":
    run_rag_evaluation()
