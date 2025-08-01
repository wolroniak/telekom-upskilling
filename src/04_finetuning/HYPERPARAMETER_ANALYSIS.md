
# Hyperparameter Sweep Analysis Guide

This document outlines the steps to analyze the results from the hyperparameter sweep and select the best-performing model for the final application.

---

## 1. Understanding the Results

After the `run_hyperparameter_sweep.py` script finishes, you will find a new directory: `models/hyperparameter_sweep/`.

Inside this directory, there is a separate sub-folder for each experimental run, named according to its parameters (e.g., `run_1_lr-5e-5_rank-16_epochs-1`). Each of these folders contains:
- The fine-tuned LoRA model adapter (`final_model`).
- The TensorBoard logs (`logs`).

---

## 2. Launching TensorBoard

TensorBoard is a visualization toolkit that allows you to compare the metrics from different training runs. To analyze our results, we will use it to view the `train_loss` for each experiment.

**Action:** Open a new terminal in the project root directory and run the following command:

```bash
tensorboard --logdir models/hyperparameter_sweep
```

This will start the TensorBoard server. You can access it by opening the provided URL (usually `http://localhost:6006/`) in your web browser.

---

## 3. Analyzing the `train_loss` in TensorBoard

In the TensorBoard interface, you will see the `train_loss` graphs for all four experimental runs plotted together.

#### What to Look For:

The `train_loss` represents how "wrong" the model's predictions were during training. A lower loss is better. When comparing the graphs, you are looking for the **best combination** of these factors:

1.  **Lowest Final Value:** The primary indicator of a good model is the one whose loss curve reaches the lowest final value. This means it has learned the training data most effectively.
2.  **Fastest Convergence:** A curve that drops quickly is more efficient. If two models reach the same low loss, the one that got there in fewer steps is often preferable.
3.  **Stability:** The curve should be relatively smooth. A very noisy or jagged loss curve can indicate unstable training.

**Decision:** Identify the run that has the most favorable loss curve (lowest, fastest, smoothest). Note down its parameters (e.g., `run_2` with `lr=2e-4` and `rank=32`).

---

## 4. (Optional but Recommended) Qualitative Evaluation

While `train_loss` is a great metric, it's always best to see how the model *actually* performs on real examples. You can use our existing `evaluation.py` script to compare your top 2-3 models from the sweep.

**Action:**
1. Open the `src/04_finetuning/evaluation.py` script.
2. Locate the `main` function.
3. Change the `adapter_path` variable to point to the `final_model` directory of the experiment you want to test. For example:
   ```python
   # Before
   adapter_path="models/Qwen3-0.6B-fine-tuned/final_model"

   # After (to test run 2 from the sweep)
   adapter_path="models/hyperparameter_sweep/run_2_lr-2e-4_rank-32_epochs-1/final_model"
   ```
4. Run the script (`python src/04_finetuning/evaluation.py`) and compare the outputs side-by-side to make your final decision.

---

## 5. Integrating the Best Model

Once you have chosen the winning set of hyperparameters, the final step is to integrate its model into our final application.

**Action:**
1. Open `src/04_finetuning/new_application/llm_agent_new.py`.
2. Find the `MODEL_CONFIG` dictionary at the top of the file.
3. Update the `adapter_path` for the `"Qwen3-0.6B-fine-tuned"` entry to point to the directory of your chosen model from the sweep.

Now, when you run `run_final_agent.py`, it will use the best-performing model from all of your experiments.
