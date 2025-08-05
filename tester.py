import os
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from autoencoder import DeepAutoencoder
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from audioDeepfakeAttribution import AudioDeepfakeAttributionModel, predict_with_confidence_threshold as infer_ADA
from audioDeepfakeModelRecognition import ADMR_model, predict_with_confidence_threshold as infer_ADMR

from testDataset import GeneralizationDataset, ErrorPropagationDataset

def compute_eer(y_true, y_score):
    """
    Compute Equal Error Rate (EER) from true labels and prediction scores.
    
    The EER is the point where False Positive Rate equals False Negative Rate,
    commonly used in biometric and audio authentication systems.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_score : array-like
        Prediction scores or probabilities
        
    Returns:
    --------
    tuple
        (eer, eer_threshold) where:
        - eer: Equal Error Rate value
        - eer_threshold: Threshold value that produces the EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold

def error_propagation_test(ADA_model, ADMR_model, test_loader, device):
    """
    Test error propagation through the LAVA framework pipeline.
    
    This function evaluates how errors from the ADA (Audio Deepfake Attribution) stage
    propagate to the ADMR (Audio Deepfake Model Recognition) stage. It simulates the
    complete pipeline workflow where ADA predictions determine whether ADMR is invoked.
    
    Pipeline Logic:
    ---------------
    1. ADA classifies samples into datasets (CodecFake=0, ASVspoof2021=1, FakeOrReal=2)
    2. ADMR is only invoked for samples classified as CodecFake (label=0)
    3. ADMR errors are only meaningful when ADA correctly identifies CodecFake samples
    
    Parameters:
    -----------
    ADA_model : torch.nn.Module
        Trained Audio Deepfake Attribution model
    ADMR_model : torch.nn.Module  
        Trained Audio Deepfake Model Recognition model
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test samples with (audio, dataset_label, generation_label)
    device : torch.device
        Device for model inference (CPU or CUDA)
        
    Returns:
    --------
    tuple
        (ada_true, ada_pred, admr_true, admr_pred, valid_mask) where:
        - ada_true/ada_pred: Lists of true/predicted ADA labels
        - admr_true/admr_pred: Lists of true/predicted ADMR labels  
        - valid_mask: Boolean mask indicating valid ADMR predictions
        
    Error Analysis:
    ---------------
    - Phase 1 errors: ADA misclassifications (all samples)
    - Phase 2 errors: ADMR misclassifications (only CodecFake samples)
    - Error propagation: How ADA errors affect ADMR performance
    """
    ADA_model.eval()
    ADMR_model.eval()

    phase1_errors = 0
    phase2_errors = 0
    total_samples = 0
    admr_samples = 0  # only samples classified as CodecFake

    ada_true, ada_pred = [], []
    admr_true, admr_pred = [], []
    valid_mask = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Error Propagation", unit="batch"):
            x, dataset_label, gen_label = batch
            x = x.to(device)
            dataset_label = dataset_label.to(device)
            gen_label = gen_label.to(device)

            # Step 1: Run both ADA and ADMR models
            ada_output = ADA_model(x)
            pred_ada = torch.argmax(ada_output, dim=1)

            admr_output = ADMR_model(x)
            pred_admr = torch.argmax(admr_output, dim=1)

            for i in range(len(x)):
                total_samples += 1
                true_ds = dataset_label[i].item()
                pred_ds = pred_ada[i].item()
                ada_true.append(true_ds)
                ada_pred.append(pred_ds)

                true_gen = gen_label[i].item()
                pred_gen = pred_admr[i].item()

                if pred_ds != true_ds:
                    # ADA classification error
                    phase1_errors += 1
                    if pred_ds == 0:
                        # Wrongly classified as CodecFake → ADMR invoked but result is invalid
                        phase2_errors += 1
                        admr_true.append(0)
                        admr_pred.append(0)
                        valid_mask.append(False)
                        admr_samples += 1
                    else:
                        # Wrong classification into non-CodecFake → ADMR not invoked
                        admr_true.append(0)
                        admr_pred.append(0)
                        valid_mask.append(False)
                elif pred_ds == 0:
                    # Correctly identified as CodecFake → run ADMR
                    admr_samples += 1
                    if pred_gen != true_gen:
                        phase2_errors += 1
                    admr_true.append(true_gen)
                    admr_pred.append(pred_gen)
                    valid_mask.append(True)
                else:
                    # Correctly classified as ASVspoof2021 or FakeOrReal → no ADMR needed
                    admr_true.append(0)
                    admr_pred.append(0)
                    valid_mask.append(False)

    print(f"Total samples processed: {total_samples}")
    print(f"Phase 1 - Audio Deepfake Attribution (ADA): {phase1_errors}/{total_samples} errors ({(phase1_errors/total_samples)*100:.2f}%)")
    print(f"Phase 2 - Audio Deepfake Model Recognition (ADMR): {phase2_errors}/{admr_samples} errors ({(phase2_errors / admr_samples)*100:.2f}%)")
    print(f"Samples passed to ADMR: {admr_samples}")

    return ada_true, ada_pred, admr_true, admr_pred, valid_mask





def plot_nested_confusion_matrix(ada_true, ada_pred, admr_true, admr_pred, valid_mask, save_path):
    """
    Plot nested confusion matrix showing both ADA and ADMR performance.
    
    Creates a combined visualization showing:
    - ADA confusion matrix (3x3) for dataset attribution
    - ADMR confusion matrix (6x6) for model recognition
    - Both matrices in a single 9x9 nested layout
    
    Parameters:
    -----------
    ada_true : list
        True ADA labels (dataset attribution)
    ada_pred : list  
        Predicted ADA labels
    admr_true : list
        True ADMR labels (model recognition)
    admr_pred : list
        Predicted ADMR labels
    valid_mask : list
        Boolean mask for valid ADMR predictions
    save_path : str
        Path to save the confusion matrix plot
        
    Matrix Layout:
    --------------
    - Positions 0-2: ADA classes (CodecFake, ASVspoof2021, FakeOrReal)
    - Positions 3-8: ADMR classes (6 CodecFake generation models)
    """
    cm_ada = confusion_matrix(ada_true, ada_pred, labels=[0, 1, 2])
    
    admr_true = np.array(admr_true, dtype=int)
    admr_pred = np.array(admr_pred, dtype=int)
    valid_mask = np.array(valid_mask, dtype=bool)
    cm_admr = confusion_matrix(admr_true[valid_mask], admr_pred[valid_mask], labels=[0, 1, 2, 3, 4, 5])

    nested_cm = np.zeros((9, 9), dtype=int)
    nested_cm[0:3, 0:3] = cm_ada
    nested_cm[3:9, 3:9] = cm_admr

    plt.figure(figsize=(10, 10))
    sns.heatmap(nested_cm, annot=True, fmt='d', cmap="Blues", xticklabels=list(range(9)), yticklabels=list(range(9)))
    plt.title("Error Propagation Matrix (0-2: ADA, 3-8: ADMR)")
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generalization_test(generalization_loader, ADA_model, ADMR_model, device):
    """
    Test model generalization on unseen datasets.
    
    Evaluates how well the trained ADA and ADMR models perform on completely
    unseen data (ASVspoof2019LA) to assess cross-dataset generalization capabilities.
    This test is crucial for understanding model robustness and real-world applicability.
    
    Parameters:
    -----------
    generalization_loader : torch.utils.data.DataLoader
        DataLoader containing unseen test samples (typically ASVspoof2019LA)
    ADA_model : torch.nn.Module
        Trained Audio Deepfake Attribution model
    ADMR_model : torch.nn.Module
        Trained Audio Deepfake Model Recognition model  
    device : torch.device
        Device for model inference (CPU or CUDA)
        
    Test Strategy:
    --------------
    - ADA Test: How well the model attributes unseen samples to known datasets
    - ADMR Test: How well the model recognizes generation patterns in unseen data
    - Focus on zero-shot performance without fine-tuning
    
    Metrics Reported:
    -----------------
    - Overall accuracy for both ADA and ADMR
    - Distribution of misclassifications across dataset categories
    - Percentage breakdown of prediction errors
    """
    ADA_model.eval()
    ADMR_model.eval()
    y_true_ADA = []
    y_pred_ADA = []

    with torch.no_grad():
        for batch in tqdm(generalization_loader, desc="Testing Generalization (ADA)", unit="batch"):
            x, ADA_label, label = batch
            x = x.to(device)
            output, _ = infer_ADA(ADA_model, x, device)
            pred = output
            y_pred_ADA.extend(np.array(pred, dtype=int))
            y_true_ADA.extend(ADA_label.numpy())

    y_true_ADMR = []
    y_pred_ADMR = []

    with torch.no_grad():
        for batch in tqdm(generalization_loader, desc="Testing Generalization (ADMR)", unit="batch"):
            x, ADA_label, label = batch
            x = x.to(device)
            output, _ = infer_ADMR(ADMR_model, x, device)
            pred = output
            y_pred_ADMR.extend(np.array(pred, dtype=int))
            y_true_ADMR.extend(label.numpy())

    hit = 0
    miss = {"0": 0, "1": 0, "2": 0}
    print("Generalization Test Metrics (ADA):")
    for i in range(0, len(y_true_ADA)):
        if y_true_ADA[i] == y_pred_ADA[i]:
            hit += 1
        else:
            miss[str(y_pred_ADA[i])] += 1
    print(f"Accuracy: {hit}/{len(y_true_ADA)} -> ({(hit/len(y_true_ADA))*100:.2f}%)")
    print_miss = {"CodecFake": miss["0"], "ASVspoof2021": miss["1"], "FakeOrReal": miss["2"]}
    print(f"Misses: {print_miss}\nPercentages: CodecFake: {print_miss['CodecFake']/(len(y_true_ADA))*100:.2f}%, ASVspoof2021: {print_miss['ASVspoof2021']/(len(y_true_ADA))*100:.2f}%, FakeOrReal: {print_miss['FakeOrReal']/(len(y_true_ADA))*100:.2f}%")

    hit = 0
    print("Generalization Test Metrics (ADMR):")
    for i in range(0, len(y_true_ADMR)):
        if y_true_ADMR[i] == y_pred_ADMR[i]:
            hit += 1
    print(f"Accuracy: {hit}/{len(y_true_ADMR)} -> ({(hit/len(y_true_ADMR))*100:.2f}%)")

if __name__ == "__main__":
    # Model paths - update these to point to your trained models
    pretrained_autoencoder_path = '/path/to/models/autoencoder.pt'
    ADA_model_path = '/path/to/models/ADA_model.pt'
    ADMR_model_path = '/path/to/models/ADMR_model.pt'

    # Dataset paths - update these to point to your processed audio data
    base_dir = "/path/to/processed/audio/data"
    real_for_dir = f"{base_dir}/FOR/real"
    real_avs_dir = f"{base_dir}/ASVspoof2021/real"
    real_codec_dir = f"{base_dir}/CodecFake/real"
    fake_for_dir = f"{base_dir}/FOR/fake"
    fake_avs_dir = f"{base_dir}/ASVspoof2021/fake"
    fake_codec_dir = f"{base_dir}/CodecFake/fake"

    # Generalization dataset (unseen data)
    fake_avs19LA_dir = f"{base_dir}/ASVspoof2019LA/fake"

    generalization_dataset = GeneralizationDataset(
        avs19LA_fake_dir=fake_avs19LA_dir
    )

    generalization_loader = DataLoader(generalization_dataset, batch_size=16, shuffle=True)

    # Error propagation dataset
    error_propagation_dataset = ErrorPropagationDataset(
        for_fake_dir=fake_for_dir,
        for_real_dir=real_for_dir,
        avs_fake_dir=fake_avs_dir,
        avs_real_dir=real_avs_dir,
        codec_real_dir=real_codec_dir,
        codec_csvfile="/path/to/csv/codecfake/test.csv"
    )

    error_propagation_loader = DataLoader(error_propagation_dataset, batch_size=16, shuffle=True)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = DeepAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(pretrained_autoencoder_path, map_location=device))
    ADA_model = AudioDeepfakeAttributionModel(pretrained_autoencoder=autoencoder).to(device)
    ADMR_model_instance = ADMR_model(pretrained_autoencoder=autoencoder).to(device)
    ADA_model.load_state_dict(torch.load(ADA_model_path, map_location=device))
    ADMR_model_instance.load_state_dict(torch.load(ADMR_model_path, map_location=device))
    print("Models loaded successfully.")
    
    # Run error propagation test
    print("Starting error propagation test...")
    ada_true, ada_pred, admr_true, admr_pred, valid_mask = error_propagation_test(ADA_model, ADMR_model_instance, error_propagation_loader, device)
    print("Error propagation test completed.")
    plot_nested_confusion_matrix(ada_true, ada_pred, admr_true, admr_pred, valid_mask, "/path/to/reports/error_propagation_test.png")
    
    # Run generalization test
    print("Starting generalization test...")
    generalization_test(generalization_loader, ADA_model, ADMR_model_instance, device)
    print("Generalization test completed.")


    
