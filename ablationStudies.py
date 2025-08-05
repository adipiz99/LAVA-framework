
import os
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from autoencoder import DeepAutoencoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PathLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading audio samples from CSV files containing paths and labels.
    
    This dataset is specifically designed for loading preprocessed audio tensors (.pt files)
    that have been saved as PyTorch tensors. It supports optional zero-based label conversion
    for compatibility with different labeling schemes.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing 'path' and 'label' columns
    zero_base : bool, optional (default=False)
        If True, converts labels to zero-based indexing by subtracting 1
        Useful when labels are 1-based but model expects 0-based
    
    CSV File Format:
    ----------------
    The CSV file should contain two columns:
    - 'path': Absolute or relative paths to .pt tensor files
    - 'label': Integer labels for classification
    
    Example CSV:
    path,label
    /path/to/sample1.pt,0
    /path/to/sample2.pt,1
    /path/to/sample3.pt,2
    
    Returns:
    --------
    tuple: (tensor, label)
        - tensor: PyTorch tensor loaded from the .pt file (converted to float)
        - label: Integer label (optionally converted to zero-based)
    
    Usage:
    ------
    # For zero-based labels (0, 1, 2, ...)
    dataset = PathLabelDataset("train.csv", zero_base=False)
    
    # For converting 1-based labels to 0-based (1,2,3,... -> 0,1,2,...)
    dataset = PathLabelDataset("train.csv", zero_base=True)
    """
    def __init__(self, csv_file, zero_base=False):
        df = pd.read_csv(csv_file)
        self.samples = [(Path(p), l) for p, l in zip(df['path'], df['label'])]
        self.zero_base = zero_base

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(Path(path).resolve()).float()
        if self.zero_base:
            label = label - 1
        return tensor, label

class ModelWithoutAttention(nn.Module):
    """
    Neural network model for audio deepfake classification without attention mechanism.
    
    This model uses a pretrained autoencoder as a feature extractor (frozen layers except
    the last layer) and adds a classification head for multi-class classification tasks.
    This architecture is used for ablation studies to compare performance with and without
    attention mechanisms.
    
    Architecture:
    -------------
    1. Pretrained Encoder (mostly frozen) - Extracts latent features from audio
    2. Adaptive Average Pooling - Reduces temporal dimension
    3. Classification Head - Fully connected layers for classification
    
    Parameters:
    -----------
    pretrained_autoencoder : DeepAutoencoder
        Pretrained autoencoder model whose encoder will be used as feature extractor
    num_classes : int
        Number of output classes for classification (e.g., 3 for ADA, 6 for ADMR)
    
    Model Components:
    -----------------
    - encoder: Frozen pretrained encoder (except last layer "10.weight", "10.bias")
    - classifier: Sequential layers for classification
        * AdaptiveAvgPool1d(1): Reduces temporal dimension to 1
        * Flatten(): Flattens the pooled features
        * Linear(256, 128): First fully connected layer with ReLU
        * Linear(128, num_classes): Output layer for classification
    
    Forward Pass:
    -------------
    Input: Audio tensor of shape [batch_size, channels, time_steps]
    1. Extract features using encoder -> [batch_size, 256, time_steps]
    2. Apply classification head -> [batch_size, num_classes]
    Output: Logits for each class
    
    Usage:
    ------
    # Load pretrained autoencoder
    autoencoder = DeepAutoencoder()
    autoencoder.load_state_dict(torch.load("autoencoder.pt"))
    
    # Create classification model
    model = ModelWithoutAttention(autoencoder, num_classes=6)
    
    # Forward pass
    logits = model(audio_tensor)
    """
    def __init__(self, pretrained_autoencoder, num_classes):
        super().__init__()
        self.encoder = pretrained_autoencoder.encoder
        for name, param in self.encoder.named_parameters():
            if name not in ["10.weight", "10.bias"]:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

def train_model(model, train_loader, val_loader, device, epochs, lr, save_path):
    """
    Trains a classification model using the provided data loaders with early stopping.
    
    This function implements a standard training loop with validation monitoring and
    automatic model saving when validation loss improves. Only trainable parameters
    are optimized (useful for transfer learning with frozen layers).
    
    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to train
    train_loader : DataLoader
        DataLoader containing training data (features, labels)
    val_loader : DataLoader
        DataLoader containing validation data for monitoring performance
    device : torch.device
        Device to run training on ('cuda' or 'cpu')
    epochs : int
        Maximum number of training epochs
    lr : float
        Learning rate for the Adam optimizer
    save_path : str
        Path where the best model state dict will be saved
    
    Training Process:
    -----------------
    1. Uses Adam optimizer on trainable parameters only
    2. CrossEntropyLoss for multi-class classification
    3. Monitors validation loss for early stopping
    4. Saves model when validation loss improves
    5. Prints training progress and validation metrics
    
    Output Files:
    -------------
    Saves the best model's state_dict to save_path when validation loss improves
    
    Printed Output:
    ---------------
    - Training loss per epoch
    - Validation classification report with precision, recall, F1-score
    - Model save confirmation when validation improves
    
    Usage:
    ------
    train_model(
        model=my_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cuda"),
        epochs=50,
        lr=1e-4,
        save_path="/path/to/models/best_model.pt"
    )
    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                val_loss += criterion(output, y).item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        print(classification_report(y_true, y_pred, digits=4))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

def evaluate(model, loader, device, report_path, num_classes):
    """
    Evaluates a trained model on test data and saves a detailed classification report.
    
    This function performs inference on the provided dataset, computes classification
    metrics, and saves a comprehensive report in CSV format for further analysis.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained neural network model to evaluate
    loader : DataLoader
        DataLoader containing test/evaluation data
    device : torch.device
        Device to run evaluation on ('cuda' or 'cpu')
    report_path : str
        Path where the classification report CSV will be saved
    num_classes : int
        Number of classes in the classification task (for validation)
    
    Evaluation Process:
    -------------------
    1. Sets model to evaluation mode (disables dropout, batchnorm training)
    2. Performs inference on all test samples without gradient computation
    3. Collects true labels and predicted labels
    4. Computes classification metrics using sklearn
    5. Saves detailed report to CSV file
    
    Output Files:
    -------------
    Creates a CSV file at report_path containing:
    - Per-class precision, recall, F1-score, support
    - Macro and weighted averages
    - Overall accuracy
    - Detailed breakdown by class
    
    Metrics Computed:
    -----------------
    - Precision: TP / (TP + FP) for each class
    - Recall: TP / (TP + FN) for each class  
    - F1-score: Harmonic mean of precision and recall
    - Support: Number of samples per class
    - Accuracy: Overall classification accuracy
    
    Usage:
    ------
    evaluate(
        model=trained_model,
        loader=test_loader,
        device=torch.device("cuda"),
        report_path="/path/to/reports/test_report.csv",
        num_classes=6
    )
    
    Example Output CSV Structure:
    -----------------------------
    ,precision,recall,f1-score,support
    0,0.8500,0.8200,0.8348,100
    1,0.9100,0.8800,0.8948,95
    ...
    accuracy,,,0.8750,595
    macro avg,0.8650,0.8400,0.8520,595
    weighted avg,0.8755,0.8750,0.8751,595
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Saved report to {report_path}")

def run_task(task_name, num_classes, csv_dir, model_path, output_dir, label_to_zero_base):
    """
    Executes a complete machine learning pipeline for audio deepfake classification tasks.
    
    This function orchestrates the entire workflow: loading data, creating model,
    training, and evaluation. It's designed for ablation studies comparing models
    without attention mechanisms on different classification tasks.
    
    Parameters:
    -----------
    task_name : str
        Name of the task for logging and file naming (e.g., "ADA", "ADMR")
    num_classes : int
        Number of output classes for the classification task
        - ADA (Audio Deepfake Attribution): 3 classes (different datasets)
        - ADMR (Audio Deepfake Model Recognition): 6 classes (different models)
    csv_dir : str
        Directory containing train.csv, val.csv, and test.csv files
    model_path : str
        Path to the pretrained autoencoder model (.pt file)
    output_dir : str
        Directory where trained model and evaluation report will be saved
    label_to_zero_base : bool
        Whether to convert labels to zero-based indexing
        - False: Keep original labels (for datasets already 0-based)
        - True: Convert 1-based to 0-based (subtract 1 from each label)
    
    Pipeline Steps:
    ---------------
    1. Load pretrained autoencoder and create classification model
    2. Load train/validation/test datasets from CSV files
    3. Create data loaders with appropriate batch size
    4. Train model with validation monitoring
    5. Load best model and evaluate on test set
    6. Save evaluation report
    
    Output Files:
    -------------
    {output_dir}/{task_name}_no_attention_model.pt - Trained model state dict
    {output_dir}/{task_name}_no_attention_report.csv - Evaluation report
    
    Expected CSV Structure:
    -----------------------
    Each CSV file should contain:
    - 'path': Path to preprocessed audio tensor (.pt file)
    - 'label': Integer class label
    
    Training Configuration:
    -----------------------
    - Batch size: 16
    - Epochs: 50
    - Learning rate: 1e-4
    - Optimizer: Adam (only trainable parameters)
    - Loss: CrossEntropyLoss
    
    Usage Examples:
    ---------------
    # Audio Deepfake Attribution (3 datasets, 0-based labels)
    run_task(
        task_name="ADA",
        num_classes=3,
        csv_dir="/path/to/csv/ADA_split",
        model_path="/path/to/models/autoencoder.pt",
        output_dir="/path/to/reports/ADA_no_attention",
        label_to_zero_base=False
    )
    
    # Audio Deepfake Model Recognition (6 models, 1-based -> 0-based)
    run_task(
        task_name="ADMR", 
        num_classes=6,
        csv_dir="/path/to/csv/ADMR_split",
        model_path="/path/to/models/autoencoder.pt", 
        output_dir="/path/to/reports/ADMR_no_attention",
        label_to_zero_base=True
    )
    """
    print(f"Running task: {task_name} with {num_classes} classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    model = ModelWithoutAttention(autoencoder, num_classes).to(device)

    train_set = PathLabelDataset(os.path.join(csv_dir, "train.csv"), zero_base=label_to_zero_base)
    val_set = PathLabelDataset(os.path.join(csv_dir, "val.csv"), zero_base=label_to_zero_base)
    test_set = PathLabelDataset(os.path.join(csv_dir, "test.csv"), zero_base=label_to_zero_base)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    model_save_path = os.path.join(output_dir, f"{task_name}_no_attention_model.pt")
    train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    report_path = os.path.join(output_dir, f"{task_name}_no_attention_report.csv")
    print(f"Evaluating model on {task_name} test set...")
    evaluate(model, test_loader, device, report_path, num_classes)

if __name__ == "__main__":
    run_task(
        task_name="ADA",
        num_classes=3,
        csv_dir="/path/to/csv/ADA_split",
        model_path="/path/to/models/autoencoder.pt",
        output_dir="/path/to/reports/ADA_no_attention",
        label_to_zero_base=False
    )
    run_task(
        task_name="ADMR",
        num_classes=6,
        csv_dir="/path/to/csv/ADMR_split",
        model_path="/path/to/models/autoencoder.pt",
        output_dir="/path/to/reports/ADMR_no_attention",
        label_to_zero_base=True
    )

    print("All tasks completed successfully.")
