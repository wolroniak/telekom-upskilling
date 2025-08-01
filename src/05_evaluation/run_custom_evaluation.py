#!/usr/bin/env python3
"""
Custom Evaluation Runner

This script runs the comprehensive evaluation of all 4 system variants
following the official task requirements.

Usage:
    python run_custom_evaluation.py
"""

import sys
from pathlib import Path

# Add the custom_evaluation directory to path
sys.path.insert(0, str(Path(__file__).parent / "custom_evaluation"))

from evaluator import main

if __name__ == "__main__":
    print("Custom LLM Agent Evaluation")
    print("Using Claude Sonnet 3.5 for LLM-based feedback on Helpfulness, Clarity, Empathy, and Safety")
    print()
    
    main()