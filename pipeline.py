import torch
from autoencoder import DeepAutoencoder
from audioDeepfakeAttribution import AudioDeepfakeAttributionModel, predict_with_confidence_threshold as predict_ADA
from audioDeepfakeModelRecognition import ADMR_model, predict_with_confidence_threshold as predict_ADMR
from pathlib import Path

def load_model(model_class, model_path, autoencoder_path=None):
    """
    Utility function to load pretrained models with optional autoencoder dependency.
    
    This function handles loading of different model types, including those that require
    a pretrained autoencoder (like ADA and ADMR models) and standalone models.
    Automatically detects and uses the best available device (CUDA/CPU).
    
    Parameters:
    -----------
    model_class : class
        The model class to instantiate (e.g., AudioDeepfakeAttributionModel, ADMR_model)
    model_path : str
        Path to the trained model weights (.pt file)
    autoencoder_path : str, optional (default=None)
        Path to pretrained autoencoder weights (.pt file)
        Required for models that use pretrained encoders (ADA, ADMR)
        
    Returns:
    --------
    tuple: (model, device)
        model : torch.nn.Module
            Loaded model in evaluation mode, ready for inference
        device : torch.device
            Device where the model is loaded ('cuda' or 'cpu')
            
    Process:
    --------
    1. Detect best available device (CUDA if available, else CPU)
    2. If autoencoder_path provided:
       - Load pretrained autoencoder
       - Create model instance with pretrained autoencoder
    3. Else: Create standalone model instance
    4. Load trained model weights
    5. Set model to evaluation mode
    
    Usage:
    ------
    # For models requiring autoencoder (ADA/ADMR)
    model, device = load_model(
        model_class=AudioDeepfakeAttributionModel,
        model_path="/path/to/models/ADA_model.pt",
        autoencoder_path="/path/to/models/autoencoder.pt"
    )
    
    # For standalone models
    model, device = load_model(
        model_class=StandaloneModel,
        model_path="/path/to/models/standalone_model.pt"
    )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if autoencoder_path:
        ae = DeepAutoencoder().to(device)
        ae.load_state_dict(torch.load(autoencoder_path, map_location=device))
        model = model_class(ae).to(device)
    else:
        model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def run_pipeline(tensor, model_ADA, model_ADMR, device, thresholds):
    """
    Executes the complete LAVA framework pipeline for audio deepfake analysis.
    
    This function implements the two-stage LAVA pipeline:
    1. Audio Deepfake Attribution (ADA) - Identifies the source dataset
    2. Audio Deepfake Model Recognition (ADMR) - Identifies the generation model
    
    The pipeline uses confidence thresholds to ensure reliable predictions,
    rejecting uncertain predictions to maintain high accuracy.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Input audio tensor to analyze [channels, time_steps]
    model_ADA : AudioDeepfakeAttributionModel
        Trained ADA model for dataset attribution
    model_ADMR : ADMR_model  
        Trained ADMR model for generation model recognition
    device : torch.device
        Device for inference ('cuda' or 'cpu')
    thresholds : list or tuple
        Confidence thresholds [ADA_threshold, ADMR_threshold]
        Values between 0.0 and 1.0
        
    Pipeline Logic:
    ---------------
    1. Stage 1 - ADA (Audio Deepfake Attribution):
       - Predict source dataset with confidence filtering
       - If confidence < threshold: Return uncertain result
       - If confident: Proceed to Stage 2
       
    2. Stage 2 - ADMR (Audio Deepfake Model Recognition):
       - Predict generation model with confidence filtering
       - Return final attribution results
    
    Returns:
    --------
    dict: Pipeline results
        {
            "ADA": int or -1,
            "ADMR": int or -1 or None
        }
        
    Result Interpretation:
    ----------------------
    ADA values:
    - 0, 1, 2: Confident dataset prediction (CodecFake, ASVspoof2021, FakeOrReal)
    - -1: Uncertain dataset prediction (below confidence threshold)
    
    ADMR values:
    - 0-5: Confident model prediction (generation models 1-6)
    - -1: Uncertain model prediction (below confidence threshold)  
    - None: ADMR not executed (ADA was uncertain)
    
    Usage:
    ------
    # Define confidence thresholds
    thresholds = [0.85, 0.90]  # [ADA_threshold, ADMR_threshold]
    
    # Run complete pipeline
    result = run_pipeline(
        tensor=audio_tensor,
        model_ADA=loaded_ada_model,
        model_ADMR=loaded_admr_model, 
        device=torch.device("cuda"),
        thresholds=thresholds
    )
    
    # Interpret results
    if result["ADA"] == -1:
        print("Uncertain dataset attribution")
    elif result["ADMR"] is None:
        print("ADMR not executed due to uncertain ADA")
    elif result["ADMR"] == -1:
        print(f"Dataset: {result['ADA']}, but uncertain model attribution")
    else:
        print(f"Dataset: {result['ADA']}, Generation Model: {result['ADMR']}")
    """
    tensor = tensor.unsqueeze(0).to(device)

    pred_ADA = predict_ADA(model_ADA, tensor, device, threshold=thresholds[0])[0]
    if pred_ADA == -1:
        return {"ADA": -1, "ADMR": None}

    pred_ADMR = predict_ADMR(model_ADMR, tensor, device, threshold=thresholds[1])[0]
    return {"ADA": pred_ADA, "ADMR": pred_ADMR}

if __name__ == "__main__":
    """
    Example usage of the LAVA framework pipeline for audio deepfake analysis.
    
    This example demonstrates how to:
    1. Load pretrained ADA and ADMR models
    2. Configure confidence thresholds
    3. Run the complete pipeline on audio samples
    4. Interpret the results
    
    Before running, ensure you have:
    - Trained ADA and ADMR models
    - Pretrained autoencoder
    - Preprocessed audio tensors (.pt files)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths to models (update these paths according to your setup)
    autoencoder_path = "/path/to/models/autoencoder.pt"
    ADA_model_path = "/path/to/models/ADA_model.pt"  
    ADMR_model_path = "/path/to/models/ADMR_model.pt"

    # Load models
    print("Loading ADA model...")
    model_ADA, _ = load_model(
        model_class=AudioDeepfakeAttributionModel,
        model_path=ADA_model_path,
        autoencoder_path=autoencoder_path
    )
    
    print("Loading ADMR model...")
    model_ADMR, _ = load_model(
        model_class=ADMR_model,
        model_path=ADMR_model_path, 
        autoencoder_path=autoencoder_path
    )

    # Configure confidence thresholds (adjust based on your requirements)
    # Higher thresholds = more conservative (higher precision, lower coverage)
    thresholds = [0.85, 0.90]  # [ADA_threshold, ADMR_threshold]
    
    # Example: Load and analyze audio sample
    test_tensor_path = "/path/to/audio/sample.pt"
    print(f"Loading test audio from: {test_tensor_path}")
    
    try:
        test_tensor = torch.load(Path(test_tensor_path))
        print(f"Audio tensor shape: {test_tensor.shape}")
        
        # Run complete LAVA pipeline
        print("Running LAVA pipeline...")
        result = run_pipeline(
            tensor=test_tensor,
            model_ADA=model_ADA,
            model_ADMR=model_ADMR,
            device=device,
            thresholds=thresholds
        )
        
        # Interpret and display results
        print("\n" + "="*50)
        print("LAVA PIPELINE RESULTS")
        print("="*50)
        
        # ADA Results
        if result["ADA"] == -1:
            print("Audio Deepfake Attribution (ADA): UNCERTAIN")
            print("   → Confidence below threshold, cannot reliably identify source dataset")
        else:
            dataset_names = ["CodecFake", "ASVspoof2021", "FakeOrReal"]
            print(f"Audio Deepfake Attribution (ADA): {dataset_names[result['ADA']]}")
        
        # ADMR Results  
        if result["ADMR"] is None:
            print("Audio Deepfake Model Recognition (ADMR): NOT EXECUTED")
            print("   → Skipped due to uncertain dataset attribution")
        elif result["ADMR"] == -1:
            print("Audio Deepfake Model Recognition (ADMR): UNCERTAIN") 
            print("   → Confidence below threshold, cannot reliably identify generation model")
        else:
            print(f"Audio Deepfake Model Recognition (ADMR): Generation Model {result['ADMR'] + 1}")

        print("="*50)
        
    except FileNotFoundError:
        print(f"Error: Audio file not found at {test_tensor_path}")
        print("Please update the path to point to a valid .pt audio tensor file")
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        
    print("\nPipeline execution completed.")
