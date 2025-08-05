import torch
import random
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
    
class ErrorPropagationDataset(Dataset):
    """
    Dataset for testing error propagation in the LAVA framework pipeline.
    
    This dataset combines samples from multiple audio deepfake datasets to test how
    errors propagate through the two-stage LAVA pipeline (ADA → ADMR). It includes
    both real and fake audio samples from different sources with appropriate labels
    for both dataset attribution (ADA) and model recognition (ADMR) tasks.
    
    The dataset is designed to evaluate:
    - How ADA errors affect ADMR performance
    - Model robustness across different data sources
    - Error propagation patterns in hierarchical classification
    
    Parameters:
    -----------
    for_fake_dir : str
        Directory containing FakeOrReal dataset fake samples (.pt files)
    for_real_dir : str
        Directory containing FakeOrReal dataset real samples (.pt files)
    avs_fake_dir : str
        Directory containing ASVspoof2021 fake samples (.pt files)
    avs_real_dir : str
        Directory containing ASVspoof2021 real samples (.pt files)
    codec_real_dir : str
        Directory containing CodecFake real samples (.pt files)
    codec_csvfile : str, optional (default="/path/to/csv/codecfake/test.csv")
        CSV file containing CodecFake fake samples with paths and labels
        
    Dataset Composition:
    --------------------
    - FakeOrReal: 2,000 real + 5,000 fake = 7,000 samples (ADA label: 2)
    - ASVspoof2021: 2,000 real + 5,000 fake = 7,000 samples (ADA label: 1)  
    - CodecFake: 2,000 real + 5,000 fake = 7,000 samples (ADA label: 0)
    - Total: 21,000 samples (6,000 real, 15,000 fake)
    
    Label Structure:
    ----------------
    Returns tuple: (tensor, ada_label, admr_label)
    
    ADA Labels (Dataset Attribution):
    - 0: CodecFake
    - 1: ASVspoof2021
    - 2: FakeOrReal
    
    ADMR Labels (Model Recognition):
    - -1: Not applicable (real audio or non-CodecFake fake)
    - 0-5: CodecFake generation models (converted from 1-6 to 0-5)
    
    Data Sampling Strategy:
    -----------------------
    - Balanced sampling across datasets (7K samples each)
    - Limited samples per category to prevent overfitting
    - Random shuffling with fixed seed for reproducibility
    
    Usage:
    ------
    dataset = ErrorPropagationDataset(
        for_fake_dir="/path/to/FOR/fake",
        for_real_dir="/path/to/FOR/real",
        avs_fake_dir="/path/to/ASVspoof2021/fake",
        avs_real_dir="/path/to/ASVspoof2021/real", 
        codec_real_dir="/path/to/CodecFake/real",
        codec_csvfile="/path/to/csv/codecfake/test.csv"
    )
    
    # Access samples
    audio_tensor, ada_label, admr_label = dataset[0]
    
    Applications:
    -------------
    - Error propagation analysis in LAVA pipeline
    - Cross-dataset generalization testing
    - Robustness evaluation of hierarchical models
    - Performance analysis under dataset distribution shifts
    """
    def __init__(self, for_fake_dir, for_real_dir, avs_fake_dir, avs_real_dir, codec_real_dir, codec_csvfile="/path/to/csv/codecfake/test.csv"):
        # Load data from all datasets
        self.samples = []
        df = pd.read_csv(codec_csvfile)
        for_fake = list(Path(for_fake_dir).glob("*.pt"))
        for_real = list(Path(for_real_dir).glob("*.pt"))
        avs_fake = list(Path(avs_fake_dir).glob("*.pt"))
        avs_real = list(Path(avs_real_dir).glob("*.pt"))
        codec_real = list(Path(codec_real_dir).glob("*.pt"))
        codec_fake = [(Path(p), l) for p, l in zip(df['path'], df['label'])]

        # Sample balanced subsets from each dataset
        for_real = for_real[:2000]
        avs_real = avs_real[:2000]
        codec_real = codec_real[:2000]
        for_fake = for_fake[:5000]
        avs_fake = avs_fake[:5000]
        codec_fake = codec_fake[:5000]
        # Total: 21,000 samples (6,000 real, 15,000 fake)

        # Create samples with (path, ada_label, admr_label) format
        self.samples += [(p, 2, -1) for p in for_real]      # FakeOrReal real
        self.samples += [(p, 2, -1) for p in for_fake]      # FakeOrReal fake
        self.samples += [(p, 1, -1) for p in avs_real]      # ASVspoof2021 real
        self.samples += [(p, 1, -1) for p in avs_fake]      # ASVspoof2021 fake
        self.samples += [(p, 0, -1) for p in codec_real]    # CodecFake real
        self.samples += [(p, 0, l) for p, l in codec_fake]  # CodecFake fake with model labels

        # Shuffle samples for random distribution
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ada_label, label = self.samples[idx]
        label = label - 1  # Convert 1-6 to 0-5 for ADMR (or keep -1 for non-applicable)
        tensor = torch.load(path)
        return tensor, ada_label, label 
    
class GeneralizationDataset(Dataset):
    """
    Dataset for testing generalization capabilities of LAVA framework models.
    
    This dataset uses ASVspoof2019LA samples (unseen during training) to evaluate
    how well trained models generalize to completely new datasets and generation 
    techniques. It focuses specifically on cross-dataset generalization performance
    by using only samples from a dataset not included in the training process.
    
    The dataset is designed to evaluate:
    - Cross-dataset generalization of ADA models
    - Robustness to new generation techniques (ASVspoof2019LA)
    - Model performance on unseen deepfake generation methods
    - Zero-shot performance on completely unseen data distributions
    
    Parameters:
    -----------
    avs19LA_fake_dir : str
        Directory containing ASVspoof2019LA fake samples (.pt files)
        Files should follow naming pattern with generation method indicators

        
    Dataset Composition:
    --------------------
    - ASVspoof2019LA: 20,000 fake samples (exclusively for generalization testing)

    
    Label Structure:
    ----------------
    Returns tuple: (tensor, dataset_label, model_label)
    
    Dataset Labels:
    - -1: ASVspoof2019LA (unseen dataset for generalization testing)

    Model Labels:
    - -1: Not applicable (consistent across all samples as generation method is not the focus)
    
    ASVspoof2019LA Label Extraction:
    --------------------------------
    Labels are extracted from filename patterns where the two characters
    after the first 'A' indicate the generation method (e.g., 'A01', 'A02').
    
    Data Sampling Strategy:
    -----------------------
    - Uses only ASVspoof2019LA fake samples (unseen dataset)
    - Single dataset focus for pure generalization testing
    - 20,000 samples for comprehensive evaluation
    - Random shuffling with fixed seed for reproducibility
    
    Usage:
    ------
    dataset = GeneralizationDataset(
        avs19LA_fake_dir="/path/to/ASVspoof2019LA/fake",
    )
    
    # Access samples
    audio_tensor, dataset_label, model_label = dataset[0]
    
    Applications:
    -------------
    - Cross-dataset generalization evaluation on completely unseen data
    - Zero-shot performance testing on ASVspoof2019LA
    - Robustness analysis for new generation techniques
    - Domain adaptation evaluation for audio deepfake detection models
    """
    def __init__(self, avs19LA_fake_dir):
        # Load samples from different datasets
        self.samples = []
        avs19LA_fake = list(Path(avs19LA_fake_dir).glob("*.pt"))
        
        # Extract labels from ASVspoof2019LA filenames (pattern: A01, A02, etc.)
        avs19LA_fake = [(p, int(p.parts[-1][1:3])) for p in avs19LA_fake if len(p.parts[-1]) > 2]

        # Sample balanced subsets for generalization testing
        avs19LA_fake = avs19LA_fake[:20000]  # sample for unseen data testing

        self.samples += [(p, -1, -1) for p, label in avs19LA_fake] # ASVspoof2019LA (unseen)

        # Shuffle samples for random distribution
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ds_label, model_label = self.samples[idx]
        tensor = torch.load(path)
        return tensor, ds_label, model_label    