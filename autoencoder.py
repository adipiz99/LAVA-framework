import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data import DataLoader
from fakeDataset import CodecFakeMultiClassDataset

class DeepAutoencoder(nn.Module):
    """
    Deep Convolutional Autoencoder for audio waveform feature learning and reconstruction.
    
    This autoencoder is designed for learning compressed representations of audio waveforms,
    particularly useful for deepfake audio analysis. The model uses 1D convolutional layers
    to process temporal audio data and learns to reconstruct the input waveforms through
    an encoder-decoder architecture.
    
    Architecture:
    -------------
    Encoder: 4 Conv1D layers with progressively increasing channels (1→32→64→128→256)
    - Each layer uses kernel_size=9, stride=2 for downsampling
    - BatchNorm1d + ReLU activation after each convolution
    - Results in compressed latent representation
    
    Decoder: 4 ConvTranspose1D layers with progressively decreasing channels (256→128→64→32→1)
    - Each layer uses kernel_size=9, stride=2 for upsampling
    - BatchNorm1d + ReLU activation (except final layer)
    - Dropout(0.3) before final layer for regularization
    - Final Tanh activation for output normalization
    
    Input/Output:
    -------------
    - Input: [batch_size, 1, sequence_length] - Single-channel audio waveforms
    - Latent: [batch_size, 256, compressed_length] - Learned feature representation
    - Output: [batch_size, 1, sequence_length] - Reconstructed waveforms
    
    Key Features:
    -------------
    - Temporal preservation through 1D convolutions
    - Progressive compression and expansion
    - Batch normalization for stable training
    - Dropout regularization to prevent overfitting
    - Tanh output activation for bounded reconstruction
    
    Applications:
    -------------
    - Feature extraction for downstream tasks (ADA, ADMR)
    - Audio compression and denoising
    - Anomaly detection in audio data
    - Transfer learning for audio classification tasks
    
    Usage:
    ------
    # Create and use autoencoder
    autoencoder = DeepAutoencoder()
    
    # Forward pass
    reconstructed = autoencoder(audio_waveforms)
    
    # Access encoder for feature extraction
    features = autoencoder.encoder(audio_waveforms)
    """
    def __init__(self):
        super(DeepAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def train(model, train_loader, val_loader, device, epochs=50, lr=1e-4, save_path='/path/to/models/autoencoder.pt'):
    """
    Trains the Deep Autoencoder with validation monitoring and early stopping.
    
    This function implements a complete training loop for autoencoder training using
    reconstruction loss. The model learns to compress and reconstruct audio waveforms,
    with automatic saving of the best model based on validation performance.
    
    Parameters:
    -----------
    model : DeepAutoencoder
        The autoencoder model to train
    train_loader : DataLoader
        DataLoader containing training audio samples
    val_loader : DataLoader
        DataLoader containing validation audio samples for monitoring
    device : torch.device
        Device for training ('cuda' or 'cpu')
    epochs : int, optional (default=50)
        Maximum number of training epochs
    lr : float, optional (default=1e-4)
        Learning rate for Adam optimizer
    save_path : str, optional
        Path where the best model weights will be saved
        
    Training Configuration:
    -----------------------
    - Optimizer: Adam with specified learning rate
    - Loss Function: SmoothL1Loss(beta=0.0001) for robust reconstruction
        * More robust to outliers than MSE
        * Smooth transition between L1 and L2 loss behavior
    - Early Stopping: Saves model when validation loss improves
    - Batch Processing: Processes audio samples in batches
    
    Training Process:
    -----------------
    1. Training Phase:
       - Set model to training mode
       - Forward pass: reconstruction = model(audio)
       - Compute reconstruction loss
       - Backpropagation and parameter updates
       
    2. Validation Phase:
       - Set model to evaluation mode (no gradients)
       - Compute validation reconstruction loss
       - Save model if validation loss improved
       
    3. Progress Monitoring:
       - Print training and validation loss each epoch
       - Save confirmation when model improves
    
    Output:
    -------
    Saves best model state dict to save_path when validation loss improves
    
    Loss Function Details:
    ----------------------
    SmoothL1Loss provides better training stability compared to MSE:
    - Behaves like L2 loss for small errors (smooth gradients)
    - Behaves like L1 loss for large errors (robust to outliers)
    - Beta=0.0001 controls the transition point
    
    Usage:
    ------
    # Train autoencoder
    train(
        model=autoencoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cuda"),
        epochs=50,
        lr=1e-4,
        save_path="/path/to/models/autoencoder.pt"
    )
    
    # Load trained model for inference or transfer learning
    autoencoder.load_state_dict(torch.load("/path/to/models/autoencoder.pt"))
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.0001)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                loss = criterion(out, x)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if os.path.exists(save_path):
                os.remove(save_path)
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = "/path/to/data/audio"

    dataset = CodecFakeMultiClassDataset(
        root_dir="/path/to/audio/CodecFake/fake"
    )

    # Split dataset into training and validation sets
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    seed = 42  # Set a seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = data.random_split(dataset, [train_len, val_len], generator=generator)

    print(f"Total samples: {total_len}")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    model = DeepAutoencoder().to(device)
    save_path = '/path/to/models/autoencoder.pt'
    print(f"Model will be saved to {save_path}")
    train(model, train_loader, val_loader, device, epochs=50, lr=1e-4, save_path=save_path)