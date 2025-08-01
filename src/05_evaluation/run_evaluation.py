
import os
import subprocess

def run_evaluation():
    print("--- Starting Comprehensive System Evaluation ---")
    print("This process will take a significant amount of time as it involves multiple LLM calls for each test case.")
    
    command = [
        "npx",
        "promptfoo",
        "eval",
        "-c", "promptfooconfig.yaml",
        "-o", "evaluation_results.json",
        "--max-concurrency", "1"
    ]
    
    evaluation_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
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
