import os
import random
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from fakeDataset import CodecFakeMultiClassDataset

SEED = 42
random.seed(SEED)

def prepare_ADA_splits():
    """
    Prepares a dataset split for Audio Deepfake Attribution (ADA) tasks by collecting samples 
    from multiple datasets, ensuring a balanced number of samples per dataset, and saving 
    the splits to CSV files.
    
    This function creates a multi-class dataset for attribution tasks where each class 
    represents a different source dataset. It samples up to MAX_SAMPLES_PER_DATASET 
    from each source and creates stratified train/validation/test splits.
    
    Parameters:
    -----------
    None (uses hardcoded configuration within the function)
    
    Configuration:
    --------------
    - dataset_roots: Dictionary mapping dataset names to their root directories
    - dataset_labels: Dictionary mapping dataset names to integer labels (0, 1, 2)
    - MAX_SAMPLES_PER_DATASET: Maximum number of samples to use from each dataset (25000)
    - output_csv_dir: Directory where CSV splits will be saved
    
    Dataset Structure Expected:
    ---------------------------
    Each dataset root should contain .pt files (PyTorch tensors) organized in subdirectories.
    The function recursively searches for all .pt files in the root directory.
    
    Output:
    -------
    Creates three CSV files in the output directory:
    - train.csv: 60% of the data for training
    - val.csv: 20% of the data for validation  
    - test.csv: 20% of the data for testing
    
    Each CSV file contains two columns:
    - 'path': Absolute path to the .pt file
    - 'label': Integer label indicating the source dataset (0, 1, or 2)
    
    The splits are stratified to ensure balanced representation of all classes.
    
    Usage:
    ------
    prepare_ADA_splits()
    
    Example Output Structure:
    -------------------------
    /path/to/csv/ADA_split/
    ├── train.csv  (60% of samples from all datasets)
    ├── val.csv    (20% of samples from all datasets)
    └── test.csv   (20% of samples from all datasets)
    """
    
    # paths
    dataset_roots = {
        "CodecFake": "/path/to/audio/CodecFake/fake",
        "ASVspoof2021": "/path/to/audio/ASVspoof2021/fake",
        "FakeOrReal": "/path/to/audio/FOR/fake"
    }

    # Labels
    dataset_labels = {
        "CodecFake": 0,
        "ASVspoof2021": 1,
        "FakeOrReal": 2
    }

    # Quotas to use for each dataset
    MAX_SAMPLES_PER_DATASET = 25000

    # Output
    output_csv_dir = "/path/to/csv/ADA_split"
    os.makedirs(output_csv_dir, exist_ok=True)

    # Collect samples from all datasets
    all_samples = []

    for name, root in dataset_roots.items():
        files = list(Path(root).rglob("*.pt"))
        files = sorted(files)  # deterministic order

        if len(files) < MAX_SAMPLES_PER_DATASET:
            print(f"Warning: {name} has only {len(files)} files. Using all.")
            selected = files
        else:
            selected = random.sample(files, MAX_SAMPLES_PER_DATASET)

        label = dataset_labels[name]
        all_samples.extend([(str(f.resolve()), label) for f in selected])
        print(f"{name}: selected {len(selected)} samples (label={label})")


    # Shuffle and split
    random.shuffle(all_samples)

    train_val, test = train_test_split(all_samples, test_size=0.2, stratify=[l for _, l in all_samples], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, stratify=[l for _, l in train_val], random_state=SEED)
    # Result: 60% train, 20% val, 20% test

    splits = {"train": train, "val": val, "test": test}

    # Saving CSV
    for name, split in splits.items():
        df = pd.DataFrame(split, columns=["path", "label"])
        csv_path = os.path.join(output_csv_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"{name} split saved to {csv_path} ({len(split)} samples)")

def prepare_ADMR_splits():
    """
    Prepares a dataset split for Audio Deepfake Model Recognition (ADMR) tasks by collecting 
    samples from the CodecFake dataset and saving the splits to CSV files.
    
    This function creates a multi-class dataset for model recognition tasks where each class 
    represents a different deepfake generation model. It uses the CodecFakeMultiClassDataset 
    to automatically handle the class assignment based on the model used to generate each sample.
    
    Parameters:
    -----------
    None (uses hardcoded configuration within the function)
    
    Configuration:
    --------------
    - output_csv_dir: Directory where CSV splits will be saved ("/path/to/csv/ADMR_split")
    - root_dir: Path to the CodecFake processed fake audio files
    - SEED: Random seed for reproducible splits (42)
    
    Output:
    -------
    Creates three CSV files in the output directory:
    - train.csv: 60% of the data for training
    - val.csv: 20% of the data for validation
    - test.csv: 20% of the data for testing
    
    Each CSV file contains two columns:
    - 'path': Absolute path to the .pt file
    - 'label': Integer label indicating the generation model (1-6, converted to 0-5)
    
    The splits are stratified to ensure balanced representation of all model classes.
    
    Usage:
    ------
    prepare_ADMR_splits()
    
    Example Output Structure:
    -------------------------
    /path/to/csv/ADMR_split/
    ├── train.csv  (60% of samples from all models)
    ├── val.csv    (20% of samples from all models)
    └── test.csv   (20% of samples from all models)
    
    Notes:
    ------
    - Labels are automatically assigned by CodecFakeMultiClassDataset based on directory structure
    - The function assumes 6 different generation models in the CodecFake dataset
    - Uses stratified splitting to maintain class balance across all splits
    - Files should follow naming pattern with F01, F02, ..., F06 for model identification
    """
    # Output
    output_csv_dir = "/path/to/csv/ADMR_split"
    os.makedirs(output_csv_dir, exist_ok=True)

    dataset = CodecFakeMultiClassDataset(
        root_dir="/path/to/audio/CodecFake/fake",
        seed=SEED
    )

    # Saving the dataset samples
    all_samples = dataset.samples

    # Splitting the dataset into train, val, and test sets
    train_val, test = train_test_split(all_samples, test_size=0.2, stratify=[lbl for _, lbl in all_samples], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, stratify=[lbl for _, lbl in train_val], random_state=SEED)
    # Result: 60% train, 20% val, 20% test

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    # Save splits to CSV files
    for name, split in splits.items():
        df = pd.DataFrame([(str(p), l) for p, l in split], columns=["path", "label"])
        save_path = os.path.join(output_csv_dir, f"{name}.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved {name} split with {len(split)} samples.")

if __name__ == "__main__":
    """
    Main execution script for preparing dataset splits for both ADA and ADMR tasks.
    
    This script executes both splitting functions sequentially:
    1. prepare_ADA_splits() - Creates splits for Audio Deepfake Attribution
    2. prepare_ADMR_splits() - Creates splits for Audio Deepfake Model Recognition
    
    Before running, ensure:
    - All dataset paths in the functions point to valid directories
    - The directories contain preprocessed .pt audio tensor files
    - Sufficient disk space for output CSV files
    
    Output:
    -------
    Creates two sets of CSV splits:
    - /path/to/csv/ADA_split/ - For dataset attribution tasks
    - /path/to/csv/ADMR_split/ - For model recognition tasks
    
    Each set contains train.csv, val.csv, and test.csv files.
    """
    print("Preparing attribution splits for ADA...")
    prepare_ADA_splits()
    print("ADA splits prepared successfully.")

    print("\nPreparing attribution splits for ADMR...")
    prepare_ADMR_splits()
    print("ADMR splits prepared successfully.")
    
    print("\nDataset splitting completed. Both ADA and ADMR splits are ready for training.")
