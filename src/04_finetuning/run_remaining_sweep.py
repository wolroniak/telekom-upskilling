
import os
import traceback
from datasets import load_from_disk
from fine_tuning import fine_tune_model

def get_remaining_experiments():
    """
    Defines the hyperparameter configurations for the remaining experiments.
    """
    experiments = [
        {
            "learning_rate": 2e-4, 
            "lora_rank": 32, 
            "num_train_epochs": 1,
            "lora_alpha": 64,
        },
        {
            "learning_rate": 2e-4, 
            "lora_rank": 16, 
            "num_train_epochs": 2,
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

def run_remaining_hyperparameter_sweep():
    """
    Loads the fresh dataset once and then iterates through the remaining
    hyperparameter sweep experiments.
    """
    dataset_path = "data/finetuning_dataset"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run data_preparation.py again.")
        return
    
    # Load the fresh dataset once
    try:
        dataset = load_from_disk(dataset_path)
        print("Successfully loaded dataset.")
    except Exception as e:
        print(f"Failed to load dataset. It might be corrupted. Error: {e}")
        return

    experiments = get_remaining_experiments()
    base_output_dir = "models/hyperparameter_sweep"
    
    # Starting experiment number from 2 since the first one was successful
    initial_experiment_num = 2 

    print(f"--- Starting Remaining Hyperparameter Sweep for {len(experiments)} Experiments ---")

    for i, params in enumerate(experiments):
        experiment_num = i + initial_experiment_num
        
        print("\n" + "="*80)
        print(f"--- Running Experiment {experiment_num}/{len(experiments) + initial_experiment_num - 1} ---")
        print(f"Parameters: {params}")
        print("="*80)

        try:
            run_name = (
                f"run_{experiment_num}_"
                f"lr-{params['learning_rate']}_"
                f"rank-{params['lora_rank']}_"
                f"epochs-{params['num_train_epochs']}"
            )
            output_dir = os.path.join(base_output_dir, run_name)
            
            fine_tune_model(
                output_dir=output_dir,
                dataset=dataset,
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
    run_remaining_hyperparameter_sweep()
