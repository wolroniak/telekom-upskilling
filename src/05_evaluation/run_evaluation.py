
import os
import subprocess

def run_evaluation():
    """
    Executes the promptfoo evaluation suite for the project.
    
    This script runs the promptfoo command-line tool, which will:
    1. Load the configuration from `promptfooconfig.yaml`.
    2. Iterate through each complaint in `complaints_with_references.json`.
    3. For each complaint, invoke each of the four provider scripts.
    4. Apply all defined tests (BLEU, BERTScore, and our custom LLM evaluator)
       to every generated response.
    5. Output the results to a new file and display a summary table.
    """
    print("--- Starting Comprehensive System Evaluation ---")
    print("This process will take a significant amount of time as it involves multiple LLM calls for each test case.")
    
    # Construct the command to run promptfoo
    # We specify the config file and an output file for the results.
    command = [
        "npx",
        "promptfoo",
        "eval",
        "-c", "promptfooconfig.yaml",
        "-o", "evaluation_results.json",
        "--max-concurrency", "1"
    ]
    
    # The evaluation directory should be the current working directory
    # so that promptfoo can find all the files referenced in the config.
    evaluation_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Run the command
        # We use `check=True` to ensure that if promptfoo fails, our script will also fail.
        subprocess.run(command, check=True, cwd=evaluation_dir, shell=True)
        
        print("\n--- Evaluation Finished Successfully ---")
        print("Results have been saved to 'src/05_evaluation/evaluation_results.json'")
        print("To view the detailed results in a web UI, you can run the following command:")
        print("promptfoo view -f src/05_evaluation/evaluation_results.json")

    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("The 'promptfoo' command was not found.")
        print("Please ensure that promptfoo is installed globally. You can install it with:")
        print("npm install -g promptfoo")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR ---")
        print(f"The promptfoo evaluation failed with exit code {e.returncode}.")
        print("Please check the output above for specific error messages from promptfoo.")

if __name__ == "__main__":
    run_evaluation()
