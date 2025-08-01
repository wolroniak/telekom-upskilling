
import os
import traceback
from fine_tuning import fine_tune_model

def get_sweep_experiments():
    """
    Defines all hyperparameter configurations for the sweep.
    Each dictionary represents one experiment run.
    """
    experiments = [
        {
            "learning_rate": 5e-5, 
            "lora_rank": 16, 
            "num_train_epochs": 1,
            "lora_alpha": 32,
        },
        {
            "learning_rate": 2e-4, 
            "lora_rank": 32, 
            "num_train_epochs": 1,
            "lora_alpha": 64,
        },
        {
            "learning_rate": 2e-4, 
            "lora_rank": 16, 
            "num_train_epochs": 2, # extra epoch
            "lora_alpha": 32,
        },
        {
            "learning_rate": 1e-4, 
            "lora_rank": 64, 
            "num_train_epochs": 1,
            "lora_alpha": 128,
        },
    ]
    return experiments

def run_hyperparameter_sweep():
    """
    Iterates through a list of experiments and runs the fine-tuning
    process for each, saving results to a unique directory.
    """
    experiments = get_sweep_experiments()
    base_output_dir = "models/hyperparameter_sweep"
    
    print(f"--- Starting Hyperparameter Sweep for {len(experiments)} Experiments ---")

    for i, params in enumerate(experiments):
        experiment_num = i + 1
        print("\n" + "="*80)
        print(f"--- Running Experiment {experiment_num}/{len(experiments)} ---")
        print(f"Parameters: {params}")
        print("="*80)

        try:
            # Create a unique output directory for this specific experiment
            run_name = (
                f"run_{experiment_num}_"
                f"lr-{params['learning_rate']}_"
                f"rank-{params['lora_rank']}_"
                f"epochs-{params['num_train_epochs']}"
            )
            output_dir = os.path.join(base_output_dir, run_name)
            
            # Run the fine-tuning process with the current set of parameters
            fine_tune_model(
                output_dir=output_dir,
                **params
            )
            
            print(f"--- Experiment {experiment_num} completed successfully. ---")

        except Exception:
            print(f"--- Experiment {experiment_num} failed! ---")
            print("Error details:")
            traceback.print_exc()
            print("Continuing to the next experiment...")

    print("\n" + "="*80)
    print("--- Hyperparameter Sweep Finished ---")
    print(f"All models and logs are saved in subdirectories inside: {base_output_dir}")
    print("You can now analyze the results using TensorBoard.")

if __name__ == "__main__":
    run_hyperparameter_sweep()
