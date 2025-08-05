import os
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import torch.utils.data as data
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from autoencoder import DeepAutoencoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PathLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed audio samples from CSV files.
    
    This dataset loads audio tensors that have been preprocessed and saved as PyTorch
    .pt files. It reads file paths and corresponding labels from a CSV file and loads
    the tensors on-demand during training/evaluation.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing 'path' and 'label' columns
        
    CSV Format Expected:
    --------------------
    path,label
    /path/to/sample1.pt,0
    /path/to/sample2.pt,1
    /path/to/sample3.pt,2
    
    Returns:
    --------
    tuple: (tensor, label)
        - tensor: PyTorch tensor of audio features (converted to float)
        - label: Integer class label
        
    Usage:
    ------
    dataset = PathLabelDataset("train.csv")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    """
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.samples = [(Path(p), l) for p, l in zip(df['path'], df['label'])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        tensor = torch.load(Path(path).resolve()).float()
        return tensor, label

class AudioDeepfakeAttributionModel(nn.Module):
    """
    Neural network model for Audio Deepfake Attribution (ADA) tasks.
    
    This model performs multi-class classification to identify which dataset
    a deepfake audio sample originates from. It uses a pretrained autoencoder
    as a feature extractor with an attention mechanism and classification head.
    
    Architecture:
    -------------
    1. Pretrained Encoder (mostly frozen) - Extracts latent features
    2. Attention Module - Conv1d + Sigmoid for feature weighting  
    3. Classification Head - Fully connected layers for final prediction
    
    Parameters:
    -----------
    pretrained_autoencoder : DeepAutoencoder
        Pretrained autoencoder whose encoder will be used as feature extractor
    num_classes : int, optional (default=3)
        Number of source datasets to classify (typically 3 for ADA)
        
    Model Components:
    -----------------
    - encoder: Pretrained encoder with frozen layers except final layer
    - attention: Conv1d(256->256) + Sigmoid for attention weights
    - classifier: AdaptiveAvgPool1d + Linear layers (256->128->num_classes)
    
    Forward Pass:
    -------------
    Input: [batch_size, channels, time_steps]
    1. Extract features: encoder(x) -> [batch_size, 256, time_steps]  
    2. Compute attention: attention(features) -> [batch_size, 256, time_steps]
    3. Apply attention: features * attention_weights
    4. Classify: classifier(attended_features) -> [batch_size, num_classes]
    
    Usage:
    ------
    # Load pretrained autoencoder
    autoencoder = DeepAutoencoder()
    autoencoder.load_state_dict(torch.load("autoencoder.pt"))
    
    # Create ADA model
    model = AudioDeepfakeAttributionModel(autoencoder, num_classes=3)
    
    # Forward pass
    logits = model(audio_batch)
    """
    def __init__(self, pretrained_autoencoder, num_classes=3):
        super().__init__()
        self.encoder = pretrained_autoencoder.encoder
        for name, param in self.encoder.named_parameters():
            if name not in ["10.weight", "10.bias"]:
                param.requires_grad = False

        self.attention = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        z = self.encoder(x)
        a = self.attention(z)
        z = z * a
        return self.classifier(z)

def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-4, save_path='/path/to/ADA_model.pt'):
    """
    Trains the Audio Deepfake Attribution model with validation monitoring and early stopping.
    
    This function implements a complete training loop with validation-based model saving.
    Only trainable parameters (unfrozen layers) are optimized, making it suitable for
    transfer learning scenarios.
    
    Parameters:
    -----------
    model : AudioDeepfakeAttributionModel
        The ADA model to train
    train_loader : DataLoader  
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data  
    device : torch.device
        Device for training ('cuda' or 'cpu')
    epochs : int, optional (default=20)
        Maximum number of training epochs
    lr : float, optional (default=1e-4)
        Learning rate for Adam optimizer
    save_path : str, optional
        Path to save the best model weights
        
    Training Configuration:
    -----------------------
    - Optimizer: Adam (only on trainable parameters)
    - Loss Function: CrossEntropyLoss  
    - Early Stopping: Based on validation loss improvement
    - Metrics: Classification report on validation set each epoch
    
    Output:
    -------
    Saves model state dict to save_path when validation loss improves
    
    Printed Information:
    --------------------
    - Training loss per epoch
    - Validation classification report (precision, recall, F1)
    - Model save confirmations
    
    Usage:
    ------
    train_model(
        model=ada_model,
        train_loader=train_loader, 
        val_loader=val_loader,
        device=torch.device("cuda"),
        epochs=50,
        lr=1e-4,
        save_path="/path/to/models/ADA_model.pt"
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

def evaluate_and_plot(model, loader, device, report_path, plot_dir):
    """
    Comprehensive evaluation of the ADA model with detailed visualizations and metrics.
    
    This function performs thorough model evaluation including classification metrics,
    confusion matrix, ROC curves, precision-recall curves, and t-SNE visualizations
    of the learned latent representations.
    
    Parameters:
    -----------
    model : AudioDeepfakeAttributionModel
        Trained ADA model to evaluate
    loader : DataLoader
        DataLoader containing evaluation data (typically test set)
    device : torch.device  
        Device for inference ('cuda' or 'cpu')
    report_path : str
        Path to save classification report CSV
    plot_dir : str
        Directory to save all visualization plots
        
    Generated Files:
    ----------------
    Classification Report:
    - {report_path}: Detailed classification metrics in CSV format
    
    Visualization Plots (saved to plot_dir):
    - confusion_matrix.png: Confusion matrix heatmap
    - roc_curves.png: ROC curves for all classes
    - pr_curves.png: Precision-Recall curves for all classes  
    - tsne_2d.png: 2D t-SNE of latent representations
    - tsne_3d.png: 3D t-SNE of latent representations
    
    Metrics Computed:
    -----------------
    - Per-class and overall accuracy
    - Precision, recall, F1-score for each class
    - Area Under ROC Curve (AUC) for each class
    - Average Precision (AP) for each class
    - Confusion matrix
    
    Visualizations:
    ---------------
    - ROC Curves: True Positive Rate vs False Positive Rate
    - PR Curves: Precision vs Recall trade-offs
    - t-SNE: Dimensionality reduction of latent features for visualization
    - Confusion Matrix: Classification accuracy breakdown
    
    Usage:
    ------
    evaluate_and_plot(
        model=trained_model,
        loader=test_loader,
        device=torch.device("cuda"), 
        report_path="/path/to/reports/ADA_test_report.csv",
        plot_dir="/path/to/plots/ADA/"
    )
    """
    model.eval()
    y_true, y_pred = [], []
    logits_list = []
    latents = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            z = model.encoder(x)
            pooled = torch.nn.functional.adaptive_avg_pool1d(z, 1).squeeze(-1)
            latents.append(pooled.cpu())
            out = model(x)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
            logits_list.append(out.cpu())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = torch.softmax(torch.cat(logits_list), dim=1).numpy()
    latents = torch.cat(latents, dim=0).numpy()

    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(report_path)

    matrix = confusion_matrix(y_true, y_pred)
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curves
    y_bin = label_binarize(y_true, classes=list(range(3)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "roc_curves.png"))
    plt.close()

    # Precision-Recall Curves
    precision, recall, ap = {}, {}, {}
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap[i] = average_precision_score(y_bin[:, i], probs[:, i])

    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(recall[i], precision[i], label=f"Class {i} AP={ap[i]:.2f}")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "pr_curves.png"))
    plt.close()

    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, random_state=SEED)
    emb_2d = tsne_2d.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=y_true, palette="tab10", legend="full")
    plt.title("t-SNE (2D) of Latent Space")
    plt.savefig(os.path.join(plot_dir, "tsne_2d.png"))
    plt.close()

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, random_state=SEED)
    emb_3d = tsne_3d.fit_transform(latents)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=y_true, cmap="tab10")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    ax.set_title("t-SNE (3D) of Latent Space")
    plt.savefig(os.path.join(plot_dir, "tsne_3d.png"))
    plt.close()

def compute_confidence_scores_with_preds(model, loader, device, output_csv="/path/to/csv/confidence_scores.csv"):
    """
    Computes confidence scores and predictions for all samples in the dataset.
    
    This function evaluates the model on provided data and extracts confidence scores
    (maximum softmax probability) along with predicted and true labels. The results
    are saved to a CSV file for further analysis and threshold optimization.
    
    Parameters:
    -----------
    model : AudioDeepfakeAttributionModel
        Trained model to evaluate
    loader : DataLoader
        DataLoader containing samples to analyze
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    output_csv : str, optional
        Path to save the confidence scores CSV file
        
    Process:
    --------
    1. Set model to evaluation mode
    2. For each batch, compute softmax probabilities
    3. Extract maximum confidence and corresponding prediction
    4. Collect true labels, predictions, and confidence scores
    5. Save results to CSV file
    
    Output CSV Format:
    ------------------
    confidence,true_label,pred_label
    0.9123,0,0
    0.7834,1,1  
    0.5621,2,1
    
    Usage:
    ------
    compute_confidence_scores_with_preds(
        model=trained_model,
        loader=train_loader,
        device=torch.device("cuda"),
        output_csv="/path/to/csv/ADA/confidence_scores.csv"
    )
    
    Applications:
    -------------
    - Threshold optimization for reliable predictions
    - Model calibration analysis
    - Uncertainty quantification
    - Performance analysis across confidence levels
    """
    model.eval()
    scores, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Computing confidence scores"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            max_confidence, pred = probs.max(dim=1)
            
            scores.extend(max_confidence.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy())
            true_labels.extend(y.numpy())

    df = pd.DataFrame({
        "confidence": scores,
        "true_label": true_labels,
        "pred_label": pred_labels
    })
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Confidence scores with predictions saved to {output_csv}")

def find_confidence_thresholds(csv_path):
    """
    Analyzes confidence scores to find optimal threshold for reliable predictions.
    
    This function reads confidence scores from a CSV file and analyzes the
    accuracy-coverage trade-off to find the best confidence threshold that
    maintains high accuracy while providing reasonable coverage.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing confidence scores, true labels, and predictions
        (generated by compute_confidence_scores_with_preds)
        
    Analysis Process:
    -----------------
    1. Load confidence scores and labels from CSV
    2. Test thresholds from 0.0 to 1.0 (500 points)
    3. For each threshold, compute:
       - Accuracy on samples above threshold
       - Coverage (percentage of samples above threshold)
    4. Find threshold with ≥80% coverage and maximum accuracy
    
    Output:
    -------
    Returns the optimal threshold value for ≥80% coverage
    
    Printed Results:
    ----------------
    - Coverage ≥80% Threshold and corresponding accuracy
    
    Metrics Explanation:
    --------------------
    - Coverage: Percentage of samples that exceed the confidence threshold
    - Accuracy: Classification accuracy on samples exceeding threshold
    - Trade-off: Higher thresholds = higher accuracy but lower coverage
    
    Usage:
    ------
    threshold = find_confidence_thresholds("/path/to/csv/confidence_scores.csv")
    
    Use Case:
    ---------
    In production, you can reject predictions below this threshold as "uncertain"
    to maintain high accuracy on accepted predictions.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    confidences = df["confidence"].values
    true_labels = df["true_label"].values
    pred_labels = df["pred_label"].values

    thresholds = np.linspace(0.0, 1.0, 500)
    best_cov = 0
    cov_thresh = 0

    cov_list = []

    for t in thresholds:
        mask = confidences >= t
        if np.sum(mask) == 0:
            continue

        preds = pred_labels[mask]
        targets = true_labels[mask]

        accuracy = np.mean(preds == targets)
        coverage = np.sum(mask) / len(true_labels)

        cov_list.append(coverage)

        if coverage >= 0.80 and accuracy > best_cov:
            best_cov = accuracy
            cov_thresh = t

    print(f"Coverage ≥80% Threshold: {cov_thresh:.4f} | Accuracy: {best_cov:.4f}")
    return cov_thresh

def predict_with_confidence_threshold(model, inputs, device, threshold):
    """
    Makes predictions with confidence-based filtering for reliable classification.
    
    This function performs inference and returns predictions only when the model's
    confidence exceeds a specified threshold. Low-confidence predictions are marked
    as uncertain (-1), allowing for more reliable deployment in production scenarios.
    
    Parameters:
    -----------
    model : AudioDeepfakeAttributionModel
        Trained model for making predictions
    inputs : torch.Tensor
        Input tensor(s) to classify [batch_size, channels, time_steps]
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    threshold : float
        Minimum confidence threshold (0.0 to 1.0)
        Predictions below this threshold return -1 (uncertain)
        
    Returns:
    --------
    tuple: (predictions, confidences)
        predictions : list
            Predicted class labels (0, 1, 2) or -1 for uncertain predictions
        confidences : numpy.ndarray
            Confidence scores (maximum softmax probabilities) for all samples
            
    Process:
    --------
    1. Set model to evaluation mode
    2. Compute softmax probabilities for inputs
    3. Extract maximum confidence and predicted class
    4. Return prediction if confidence ≥ threshold, else -1
    
    Usage:
    ------
    # Using threshold found by find_confidence_thresholds()
    predictions, confidences = predict_with_confidence_threshold(
        model=trained_model,
        inputs=audio_batch,
        device=torch.device("cuda"),
        threshold=0.85
    )
    
    # Handle results
    for pred, conf in zip(predictions, confidences):
        if pred == -1:
            print(f"Uncertain prediction (confidence: {conf:.3f})")
        else:
            print(f"Predicted class: {pred} (confidence: {conf:.3f})")
            
    Applications:
    -------------
    - Production deployment with uncertainty handling
    - Quality control in automated systems
    - Human-in-the-loop workflows for uncertain cases
    - Maintaining high precision in critical applications
    """
    model.eval()
    inputs = inputs.to(device)
    
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        max_confidence, predicted_class = torch.max(probs, dim=1)

        predictions = []
        confidences = max_confidence.cpu().numpy()

        for conf, pred in zip(confidences, predicted_class.cpu().numpy()):
            if conf >= threshold:
                predictions.append(pred)
            else:
                predictions.append(-1)  # -1 indicates uncertainty

    return predictions, confidences

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_csv = "/path/to/csv/ADA_dataset_split"
    model_path = "/path/to/models/autoencoder.pt"
    model_save_path = "/path/to/models/ADA_model.pt"

    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))
    model = AudioDeepfakeAttributionModel(autoencoder).to(device)

    train_set = PathLabelDataset(os.path.join(base_csv, "train.csv"))
    val_set = PathLabelDataset(os.path.join(base_csv, "val.csv"))
    test_set = PathLabelDataset(os.path.join(base_csv, "test.csv"))

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, save_path=model_save_path)
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    compute_confidence_scores_with_preds(model, train_loader, device,
                                         output_csv="/path/to/csv/ADA/confidence_scores.csv")

    find_confidence_thresholds("/path/to/csv/ADA/confidence_scores.csv")
