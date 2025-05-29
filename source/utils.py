# source/utils.py
import torch
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def add_ones(data):
    data.x = torch.ones(data.num_nodes, 1, dtype=torch.float)
    return data

def save_predictions(predictions, test_path, dataset_id):
    script_dir = os.getcwd() 
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(dataset_id)
    
    os.makedirs(submission_folder, exist_ok=True)
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, train_f1s,
                           val_losses, val_accuracies, val_f1s,
                           output_dir, plot_title_prefix=""): # Added missing val args and F1
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5)) # Wider for 3 plots, or adjust as needed

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss', color='tab:blue')
    if val_losses: # Check if validation metrics are provided
        plt.plot(epochs_range, val_losses, label='Validation Loss', color='tab:orange', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{plot_title_prefix} Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)


    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='tab:green')
    if val_accuracies:
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='tab:red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{plot_title_prefix} Accuracy')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)


    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1 Score', color='tab:purple')
    if val_f1s:
        plt.plot(epochs_range, val_f1s, label='Validation F1 Score', color='tab:brown', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'{plot_title_prefix} F1 Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)


    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
    # Sanitize plot_title_prefix for filename
    safe_title_prefix = "".join(c if c.isalnum() else "_" for c in plot_title_prefix)
    save_path = os.path.join(output_dir, f"{safe_title_prefix}_training_progress.png")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close the plot to free memory
    print(f"Saved training plot to {save_path}")

def get_dataset_id_from_path(path_str):
    if not path_str:
        print("Warning: Path string is empty, cannot extract dataset ID.")
        return "unknown"

    # Example paths:
    # "../../../../kaggle/input/dataset/A/train.json"
    # "some/path/to/B/my_data.json.gz"
    # "/absolute/path/C/another_file.json"
    # "D/only_id_and_file.json"

    path_segments = path_str.split(os.sep) # Use os.sep for platform independence
    possible_ids = {'A', 'B', 'C', 'D'} # Use a set for efficient lookup

    for i, segment in enumerate(path_segments):
        if segment in possible_ids:
            # Check if there is a next segment (the potential json file)
            if i + 1 < len(path_segments):
                next_segment = path_segments[i + 1]
                # Check if the next segment is a .json or .json.gz file
                if next_segment.endswith(".json") or next_segment.endswith(".json.gz"):
                    # We found the pattern /{A,B,C,D}/*.json(.gz)
                    return segment # This segment is the dataset ID
            # If 'segment' is an ID but there's no next segment, or the next isn't a JSON,
            # it's not the pattern we're looking for, so continue searching.
    
    # If the loop completes without returning, the pattern was not found
    print(f"Warning: Could not find pattern '/{{A,B,C,D}}/*.json(.gz)' in path '{path_str}'. Defaulting to 'unknown'.")
    return "unknown"

def get_user_input(prompt, default=None, required=False, type_cast=str):

    while True:
        user_input = input(f"{prompt} [{default}]: ")
        
        if user_input == "" and required:
            print("This field is required. Please enter a value.")
            continue
        
        if user_input == "" and default is not None:
            return default
        
        if user_input == "" and not required:
            return None
        
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_cast.__name__}.")

def get_arguments():
    args = {}
    args['train_path'] = get_user_input("Path to the training dataset (optional)")
    args['test_path'] = get_user_input("Path to the test dataset", required=True)
    
    return argparse.Namespace(**args)

def get_dataset_type():
    args = {}
    args['dataset_id'] = get_user_input("Dataset ID (A,B,C,D)", required=True)
    
    return argparse.Namespace(**args)

def populate_args(args):
    print("Arguments received:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
        
def is_gzipped_folder(path_str: str) -> bool:
    """
    Checks if the given path represents a "gzipped folder".

    This can mean one of two things:
    1. The path itself is a file ending with '.gz' (e.g., an archive like 'data.tar.gz'
       or a single gzipped file like 'log.txt.gz').
    2. The path is a directory, and ALL regular files directly within it
       have the '.gz' extension. Subdirectories are ignored in this check,
       but there must be at least one file in the directory for it to be
       considered a "gzipped folder" under this rule. An empty directory
       or a directory containing only subdirectories (no files) is NOT
       considered a gzipped folder by this rule.

    Args:
        path_str (str): The path to check.

    Returns:
        bool: True if the path represents a gzipped folder/file as defined, False otherwise.
    """
    # Normalize path (handles '~', relative paths, etc.)
    # os.path.abspath also implicitly calls os.path.normpath
    normalized_path = os.path.abspath(os.path.expanduser(path_str))

    if not os.path.exists(normalized_path):
        return False

    # Case 1: The path itself is a .gz file
    if os.path.isfile(normalized_path):
        return normalized_path.lower().endswith(".gz")

    # Case 2: The path is a directory
    if os.path.isdir(normalized_path):
        try:
            items = os.listdir(normalized_path)
        except OSError: # Handles potential permission errors, etc.
            return False # Cannot determine, so treat as not gzipped

        if not items:
            return False # Empty directory

        files_in_dir = []
        for item_name in items:
            item_path = os.path.join(normalized_path, item_name)
            if os.path.isfile(item_path): # Only consider regular files
                files_in_dir.append(item_name)

        if not files_in_dir:
            # Directory contains no regular files (e.g., only subdirectories)
            return False

        # Check if all regular files in the directory are .gz files
        return all(f.lower().endswith(".gz") for f in files_in_dir)

    # Path exists but is neither a file nor a directory (e.g., broken symlink, FIFO, socket, etc.)
    return False