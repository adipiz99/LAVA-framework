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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from autoencoder import DeepAutoencoder
from fakeDataset import CodecFakeMultiClassDataset

SEED = 42  # Set a seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Datset to load the samples from a CSV file
class PathLabelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading preprocessed audio samples from CSV files with automatic label conversion.
    
    This dataset loads audio tensors from .pt files and converts 1-based labels to 0-based labels
    automatically, making it suitable for ADMR tasks where labels typically range from 1-6 but
    PyTorch models expect 0-5.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing 'path' and 'label' columns
        
    CSV Format Expected:
    --------------------
    path,label
    /path/to/sample1.pt,1
    /path/to/sample2.pt,2
    /path/to/sample3.pt,6
    
    Label Conversion:
    -----------------
    Automatically converts 1-based labels to 0-based:
    - Input label 1 → Output label 0
    - Input label 2 → Output label 1
    - Input label 6 → Output label 5
    
    Returns:
    --------
    tuple: (tensor, label)
        - tensor: PyTorch tensor of audio features (converted to float)
        - label: Integer class label (converted to zero-based indexing)
        
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
        return tensor, label - 1  # Convert to zero-based index
    

class ADMR_model(nn.Module):
    """
    Neural network model for Audio Deepfake Model Recognition (ADMR) tasks.
    
    This model performs multi-class classification to identify which deepfake generation
    model was used to create an audio sample. It uses a pretrained autoencoder as a
    feature extractor with an attention mechanism and classification head for 6-class
    classification (different generation models).
    
    Architecture:
    -------------
    1. Pretrained Encoder (mostly frozen) - Extracts latent audio features
    2. Attention Module - Conv1d + Sigmoid for attention-based feature weighting
    3. Classification Head - Fully connected layers for model recognition
    
    Parameters:
    -----------
    pretrained_autoencoder : DeepAutoencoder
        Pretrained autoencoder whose encoder will be used as feature extractor
        
    Model Components:
    -----------------
    - encoder: Pretrained encoder with frozen layers except final layer ("10.weight", "10.bias")
    - attention: Conv1d(256->256) + Sigmoid for computing attention weights
    - classifier: Sequential layers for 6-class classification
        * AdaptiveAvgPool1d(1): Reduces temporal dimension
        * Flatten(): Flattens pooled features
        * Linear(256, 128): First FC layer with ReLU activation
        * Linear(128, 6): Output layer for 6 generation models
    
    Forward Pass:
    -------------
    Input: [batch_size, channels, time_steps]
    1. Extract features: encoder(x) -> [batch_size, 256, time_steps]
    2. Compute attention: attention(features) -> [batch_size, 256, time_steps]  
    3. Apply attention: features * attention_weights (element-wise)
    4. Classify: classifier(attended_features) -> [batch_size, 6]
    
    Usage:
    ------
    # Load pretrained autoencoder
    autoencoder = DeepAutoencoder()
    autoencoder.load_state_dict(torch.load("autoencoder.pt"))
    
    # Create ADMR model for 6 generation models
    model = ADMR_model(pretrained_autoencoder=autoencoder)
    
    # Forward pass
    logits = model(audio_batch)  # Returns logits for 6 classes
    """
    def __init__(self, pretrained_autoencoder):
        super(ADMR_model, self).__init__()
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
            nn.Linear(128, 6)  # 6 classes
        )

    def forward(self, x):
        z = self.encoder(x)  # [B, 256, T]
        a = self.attention(z)  # [B, 256, T]
        z = z * a  # Apply attention
        out = self.classifier(z)
        return out
    
    def load_ADMR_model(autoencoder_path, ADMR_model_path, device):
        """
        Static method to load a pretrained ADMR model from disk.
        
        This utility function loads both the pretrained autoencoder and the trained
        ADMR model weights, creating a complete model ready for inference.
        
        Parameters:
        -----------
        autoencoder_path : str
            Path to the pretrained autoencoder .pt file
        ADMR_model_path : str
            Path to the trained ADMR model .pt file
        device : torch.device
            Device to load the model on ('cuda' or 'cpu')
            
        Returns:
        --------
        ADMR_model
            Complete ADMR model loaded and ready for inference (in eval mode)
            
        Usage:
        ------
        device = torch.device("cuda")
        model = ADMR_model.load_ADMR_model(
            autoencoder_path="/path/to/autoencoder.pt",
            ADMR_model_path="/path/to/ADMR_model.pt", 
            device=device
        )
        
        # Now ready for inference
        predictions = model(audio_batch)
        """
        # Loads the pretrained autoencoder
        autoencoder = DeepAutoencoder().to(device)
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))

        # Loads the pretrained ADMR model
        model = ADMR_model(pretrained_autoencoder=autoencoder).to(device)
        model.load_state_dict(torch.load(ADMR_model_path, map_location=device))
        model.eval()

        print(f"ADMR model loaded from {ADMR_model_path}")
        return model

    
def train_ADMR_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, save_path='/path/to/models/ADMR_model.pt'):
    """
    Trains the ADMR model for deepfake generation model recognition with validation monitoring.
    
    This function implements a complete training loop with validation-based early stopping
    and automatic model saving. Only trainable parameters (unfrozen layers) are optimized,
    making it suitable for transfer learning from pretrained autoencoders.
    
    Parameters:
    -----------
    model : ADMR_model
        The ADMR model to train
    train_loader : DataLoader
        DataLoader containing training data (features, labels)
    val_loader : DataLoader  
        DataLoader containing validation data for performance monitoring
    device : torch.device
        Device for training ('cuda' or 'cpu')
    epochs : int, optional (default=10)
        Maximum number of training epochs
    lr : float, optional (default=1e-4)
        Learning rate for Adam optimizer
    save_path : str, optional
        Path where the best model weights will be saved
        
    Training Configuration:
    -----------------------
    - Optimizer: Adam (only on trainable parameters)
    - Loss Function: CrossEntropyLoss for 6-class classification
    - Early Stopping: Saves model when validation loss improves
    - Metrics: Detailed classification report per epoch
    
    Training Process:
    -----------------
    1. Train for one epoch, compute average training loss
    2. Evaluate on validation set, compute validation loss and metrics
    3. Save model if validation loss improved
    4. Print progress and classification report
    5. Repeat until epochs completed
    
    Output:
    -------
    Saves best model state dict to save_path when validation loss improves
    
    Printed Information:
    --------------------
    - Training and validation loss per epoch
    - Detailed classification report on validation set
    - Model save confirmations with file path
    
    Usage:
    ------
    train_ADMR_model(
        model=admr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cuda"),
        epochs=50,
        lr=1e-4,
        save_path="/path/to/models/ADMR_model.pt"
    )
    """
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(classification_report(y_true, y_pred, digits=4))

def evaluate_model(model, test_loader, device, report_path="/path/to/reports/ADMR/test_report.csv", plot_path="/path/to/plots/ADMR/"):
    """
    Comprehensive evaluation of the ADMR model with detailed visualizations and metrics.
    
    This function performs thorough model evaluation including classification metrics,
    confusion matrix, ROC curves, precision-recall curves, and t-SNE visualizations
    of the learned latent representations for all 6 generation models.
    
    Parameters:
    -----------
    model : ADMR_model
        Trained ADMR model to evaluate
    test_loader : DataLoader
        DataLoader containing test data for evaluation
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    report_path : str, optional
        Path to save detailed classification report CSV
    plot_path : str, optional  
        Directory path to save all visualization plots
        
    Generated Files:
    ----------------
    Classification Report:
    - {report_path}: Detailed per-class metrics in CSV format
    
    Visualization Plots (saved to plot_path):
    - confusion_matrix.png: 6x6 confusion matrix heatmap with class labels C1-C6
    - tsne_2d.png: 2D t-SNE visualization of latent feature space
    - tsne_3d.png: 3D t-SNE visualization of latent feature space  
    - roc_curves.png: ROC curves for all 6 classes with AUC scores
    - pr_curves.png: Precision-Recall curves for all 6 classes with AP scores
    
    Metrics Computed:
    -----------------
    - Overall test accuracy
    - Per-class precision, recall, F1-score, support
    - Confusion matrix (6x6 for generation models)
    - ROC curves and AUC for each class
    - Precision-Recall curves and Average Precision (AP) for each class
    - t-SNE embeddings for latent space visualization
    
    Evaluation Process:
    -------------------
    1. Extract latent features from encoder for t-SNE visualization
    2. Compute predictions and collect true labels
    3. Calculate all classification metrics
    4. Generate and save all visualization plots
    5. Save detailed classification report
    
    Usage:
    ------
    evaluate_model(
        model=trained_admr_model,
        test_loader=test_loader,
        device=torch.device("cuda"),
        report_path="/path/to/reports/ADMR/test_report.csv",
        plot_path="/path/to/plots/ADMR/"
    )
    
    Output Interpretation:
    ----------------------
    - Classes C1-C6 represent different deepfake generation models
    - Higher AUC values indicate better discrimination for that model
    - t-SNE plots show how well the model separates different generation models
    - Confusion matrix reveals which models are most often confused
    """
    model.eval()
    y_true, y_pred = [], []
    logits_list = []
    latent_features = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating on test set"):
            x, y = x.to(device), y.to(device)
            z = model.encoder(x)  # shape: [B, 256, T]
            pooled = torch.nn.functional.adaptive_avg_pool1d(z, 1).squeeze(-1)  # shape: [B, 256]
            latent_features.append(pooled.cpu())

            logits = model(x)
            preds = logits.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            logits_list.append(logits.cpu())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    logits_all = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits_all, dim=1).numpy()
    latents = torch.cat(latent_features, dim=0).numpy()

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    # Print and save classification report
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    # Confusion matrix
    cm_path = os.path.join(plot_path, "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f"C{i+1}" for i in range(6)], yticklabels=[f"C{i+1}" for i in range(6)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # --- t-SNE 2D ---
    tsne_2d_path = os.path.join(plot_path, "tsne_2d.png")
    tsne_2d = TSNE(n_components=2, random_state=SEED)
    emb_2d = tsne_2d.fit_transform(latents)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=y_true, palette="tab10", legend="full")
    plt.title("t-SNE (2D) of Test Latent Representations")
    plt.savefig(tsne_2d_path)
    plt.close()
    print(f"t-SNE 2D plot saved to {tsne_2d_path}")

    # --- t-SNE 3D ---
    tsne_3d_path = os.path.join(plot_path, "tsne_3d.png")
    tsne_3d = TSNE(n_components=3, random_state=SEED)
    emb_3d = tsne_3d.fit_transform(latents)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=y_true, cmap="tab10")
    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)
    ax.set_title("t-SNE (3D) of Test Latent Representations")
    plt.savefig(tsne_3d_path)
    plt.close()
    print(f"t-SNE 3D plot saved to {tsne_3d_path}")

    # --- ROC & PR Curves ---
    y_bin = label_binarize(y_true, classes=list(range(6)))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    precision, recall, ap = dict(), dict(), dict()

    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ap[i] = average_precision_score(y_bin[:, i], probs[:, i])

    # ROC Curve
    roc_path = "/path/to/plots/ADMR/roc_curves.png"
    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.plot(fpr[i], tpr[i], label=f"C{i+1} (AUC={roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curves saved to {roc_path}")

    # PR Curve
    pr_path = "/path/to/plots/ADMR/pr_curves.png"
    plt.figure(figsize=(8, 6))
    for i in range(6):
        plt.plot(recall[i], precision[i], label=f"C{i+1} (AP={ap[i]:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.savefig(pr_path)
    plt.close()
    print(f"Precision-Recall curves saved to {pr_path}")


def compute_confidence_scores_with_preds(model, loader, device, output_csv="/path/to/csv/ADMR/confidence_scores.csv"):
    """
    Computes confidence scores and predictions for ADMR model evaluation and calibration.
    
    This function evaluates the ADMR model on provided data and extracts confidence scores
    (maximum softmax probability) along with predicted and true labels. Results are saved
    for threshold optimization and model calibration analysis.
    
    Parameters:
    -----------
    model : ADMR_model
        Trained ADMR model to evaluate
    loader : DataLoader
        DataLoader containing samples to analyze (typically training set)
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    output_csv : str, optional
        Path to save the confidence scores CSV file
        
    Process:
    --------
    1. Set model to evaluation mode (disable dropout/batchnorm training)
    2. For each batch, compute softmax probabilities over 6 classes
    3. Extract maximum confidence score and corresponding prediction
    4. Collect true labels (0-5), predictions (0-5), and confidence scores
    5. Save all results to structured CSV file
    
    Output CSV Format:
    ------------------
    confidence,true_label,pred_label
    0.9456,0,0
    0.8234,1,1
    0.6789,2,1
    0.9876,5,5
    
    CSV Columns:
    ------------
    - confidence: Maximum softmax probability (0.0 to 1.0)
    - true_label: Ground truth class (0-5 for 6 generation models)
    - pred_label: Model prediction (0-5 for 6 generation models)
    
    Usage:
    ------
    compute_confidence_scores_with_preds(
        model=trained_admr_model,
        loader=train_loader,
        device=torch.device("cuda"),
        output_csv="/path/to/csv/ADMR/confidence_scores.csv"
    )
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
    Analyzes confidence scores to find optimal threshold for reliable ADMR predictions.
    
    This function reads confidence scores from CSV and analyzes the accuracy-coverage
    trade-off to find the best confidence threshold that maintains high accuracy while
    providing reasonable coverage for generation model recognition tasks.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file containing confidence scores, true labels, and predictions
        (generated by compute_confidence_scores_with_preds)
        
    Analysis Process:
    -----------------
    1. Load confidence scores and labels from CSV file
    2. Test 500 threshold values from 0.0 to 1.0
    3. For each threshold, compute:
       - Accuracy on samples exceeding the threshold
       - Coverage (percentage of samples exceeding threshold)
    4. Find optimal threshold with ≥80% coverage and maximum accuracy
    
    Metrics Explanation:
    --------------------
    - Coverage: Percentage of test samples that exceed confidence threshold
    - Accuracy: Classification accuracy on samples exceeding threshold
    - Trade-off: Higher thresholds → higher accuracy but lower coverage
    
    Output:
    -------
    Returns the optimal confidence threshold for ≥80% coverage
    
    Printed Results:
    ----------------
    - Coverage ≥80% Threshold value and corresponding accuracy
    
    Usage:
    ------
    optimal_threshold = find_confidence_thresholds(
        "/path/to/csv/ADMR/confidence_scores.csv"
    )
    
    Production Use:
    ---------------
    In production systems, predictions below this threshold can be:
    - Flagged for human review
    - Rejected as uncertain
    - Processed with alternative methods
    - Used to trigger additional verification steps
    
    ADMR-Specific Considerations:
    -----------------------------
    - Different generation models may have different confidence patterns
    - Some models might be inherently harder to distinguish
    - Threshold should balance reliability with practical coverage needs
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

def predict_with_confidence_threshold(model, inputs, device, threshold=0.95): #0.8
    """
    Makes ADMR predictions with confidence-based filtering for reliable generation model identification.
    
    This function performs inference on audio samples and returns predictions only when
    the model's confidence exceeds a specified threshold. Low-confidence predictions are
    marked as uncertain (-1), enabling reliable deployment in production scenarios where
    accurate generation model identification is critical.
    
    Parameters:
    -----------
    model : ADMR_model
        Trained ADMR model for generation model recognition
    inputs : torch.Tensor
        Input audio tensor(s) to classify [batch_size, channels, time_steps]
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    threshold : float, optional (default=0.95)
        Minimum confidence threshold (0.0 to 1.0)
        Predictions below this threshold return -1 (uncertain)
        Default 0.95 is quite conservative for high-precision applications
        
    Returns:
    --------
    tuple: (predictions, confidences)
        predictions : list
            Predicted generation model labels:
            - 0-5: Confident predictions for models 1-6
            - -1: Uncertain predictions below threshold
        confidences : numpy.ndarray
            Confidence scores (maximum softmax probabilities) for all samples
            
    Process:
    --------
    1. Set model to evaluation mode
    2. Compute softmax probabilities over 6 generation models
    3. Extract maximum confidence and predicted model class
    4. Return prediction if confidence ≥ threshold, else -1 (uncertain)
    
    Usage:
    ------
    # Using optimized threshold from find_confidence_thresholds()
    predictions, confidences = predict_with_confidence_threshold(
        model=trained_admr_model,
        inputs=audio_batch,
        device=torch.device("cuda"),
        threshold=0.85  # From threshold optimization
    )

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

# --- Execution ---
if __name__ == "__main__":
    model_path = "/path/to/models/autoencoder.pt"
    csv_path = "/path/to/csv/codecfake/fakes"
    model_save_path = "/path/to/models/ADMR_model.pt"
    generator = torch.Generator().manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(model_path, map_location=device))

    model = ADMR_model(pretrained_autoencoder=autoencoder).to(device)
        
    train_set = PathLabelDataset(os.path.join(csv_path, "train.csv"))
    val_set = PathLabelDataset(os.path.join(csv_path, "val.csv"))
    test_set = PathLabelDataset(os.path.join(csv_path, "test.csv"))

    print(f"Train set size: {len(train_set)}\nValidation set size: {len(val_set)}\nTest set size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    train_ADMR_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, save_path=model_save_path)
    print(f"ADMR model trained and saved to {model_save_path}")


    model.load_state_dict(torch.load(model_save_path, map_location=device))

    test_loader = DataLoader(test_set, batch_size=16)
    evaluate_model(model, test_loader, device, report_path="/path/to/reports/ADMR/test_report.csv", plot_path="/path/to/plots/ADMR/")
    print("Model evaluation completed.")

    compute_confidence_scores_with_preds(model, train_loader, device)
    find_confidence_thresholds("/path/to/csv/ADMR/confidence_scores.csv")

