import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "custom_evaluation"))

from evaluator import main

if __name__ == "__main__":
    print("Custom LLM Agent Evaluation")
    print("Using Claude Sonnet 3.5 for LLM-based feedback on Helpfulness, Clarity, Empathy, and Safety")
    print()
    
    main()