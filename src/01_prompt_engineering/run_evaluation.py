import yaml
import subprocess
import os

# --- Configuration ---
EVALUATION_CONFIG = {
    "models": [
        "Qwen/Qwen3-0.6B",
        "google/gemma-3-1b-it",
    ],
    "experiments": [
        {
            "name": "empathetic",
            "prompt_file": "prompts/empathetic_prompt.txt",
            "assertions": [
                {
                    "type": "llm-rubric",
                    "value": "Does the response start by clearly acknowledging the user's feelings (e.g., frustration, concern)?",
                }
            ],
        },
        {
            "name": "structured",
            "prompt_file": "prompts/structured_prompt.txt",
            "assertions": [
                {
                    "type": "llm-rubric",
                    "value": "Does the response provide a clear, numbered list of actions or points?",
                }
            ],
        },
        {
            "name": "friendly",
            "prompt_file": "prompts/friendly_prompt.txt",
            "assertions": [
                {
                    "type": "llm-rubric",
                    "value": "Is the tone of the response warm, positive, and conversational?",
                }
            ],
        },
    ],
    "common_assertions": [
        {
            "type": "llm-rubric",
            "value": "Is the response useful and does it provide a clear next step for the customer?",
        }
    ],
    "grading_provider": "anthropic:messages:claude-3-5-sonnet-20241022",
    "tests_file": "src/01_prompt_engineering/complaints.json",
    "results_dir": "results",
}

def create_promptfoo_config(model, experiment):
    """Dynamically creates a promptfoo configuration dictionary."""
    
    # Combine specific and common assertions
    all_assertions = experiment["assertions"] + EVALUATION_CONFIG["common_assertions"]
    
    # Add the grading provider to each assertion
    for assertion in all_assertions:
        assertion["provider"] = EVALUATION_CONFIG["grading_provider"]

    return {
        "prompts": [experiment["prompt_file"]],
        "providers": [
            {
                "id": "python:prompts/promptfoo_provider.py:call_agent",
                "config": {"model": model},
            }
        ],
        "tests": EVALUATION_CONFIG["tests_file"],
        "defaultTest": {"assert": all_assertions},
    }

def run_evaluation():
    """Runs the promptfoo evaluation for all models and experiments."""
    
    os.makedirs(EVALUATION_CONFIG["results_dir"], exist_ok=True)

    for model in EVALUATION_CONFIG["models"]:
        for experiment in EVALUATION_CONFIG["experiments"]:
            model_name_safe = model.replace("/", "_")
            exp_name = experiment["name"]
            
            print(f"--- Running evaluation for model: {model} with prompt: {exp_name} ---")

            config = create_promptfoo_config(model, experiment)
            
            # dynamic config to temporary file
            temp_config_path = "temp_promptfooconfig.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            output_path = os.path.join(
                EVALUATION_CONFIG["results_dir"],
                f"results_{model_name_safe}_{exp_name}.json"
            )

            # promptfoo command
            command = [
                "npx", "promptfoo", "eval",
                "-c", temp_config_path,
                "-o", output_path,
                "--max-concurrency", "1"
            ]
            
            try:
                subprocess.run(command, check=True, shell=True)
                print(f"✔ Evaluation successful. Results saved to {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"✖ Evaluation failed for model: {model}, prompt: {exp_name}")
                print(e)
            finally:
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)

if __name__ == "__main__":
    run_evaluation()
