import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD_PERCENTILE = 95  # Percentile of reconstruction errors to classify as anomalies

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

# Load and preprocess data
def load_preprocess_data(file_path='data_with_word2vec_embeddings_expanded.csv'):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Remove 'date' and 'time' columns
    df = df.drop(['date', 'time'], axis=1)
    
    # Check if the data has labels (for evaluation)
    has_labels = 'is_anomaly' in df.columns
    
    if has_labels:
        # If we have ground truth anomaly labels
        y = df['is_anomaly'].values
        X = df.drop('is_anomaly', axis=1)
    else:
        # No labels, we'll use reconstruction error to detect anomalies
        y = None
        X = df
    
    # Check for NaN values
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in the dataset")
        
        # Show columns with NaN values
        nan_columns = X.columns[X.isna().any()].tolist()
        print(f"Columns with NaN values: {nan_columns}")
        
        # Fill NaN values - use median for numeric columns and most frequent for categorical
        for col in X.columns:
            if X[col].dtype.kind in 'fc':  # float or complex
                X[col] = X[col].fillna(X[col].median())
            elif X[col].dtype.kind == 'i':  # integer
                X[col] = X[col].fillna(0)
            else:  # object or categorical
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "UNKNOWN")
    
    # Convert categorical columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Final check for any remaining NaN or Inf values
    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        print("Warning: Data still contains NaN or Inf values after preprocessing")
        # Replace any remaining NaN/Inf with zeros
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Data shape after preprocessing: {X_scaled.shape}")
    return X_scaled, y, scaler

# Train the autoencoder
def train_autoencoder(X, y=None, test_size=0.2, device=DEVICE):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y if y is not None else np.zeros(X.shape[0]), 
        test_size=test_size, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Final verification that tensors don't contain NaN/Inf values
    if torch.isnan(X_train_tensor).any() or torch.isinf(X_train_tensor).any():
        print("Warning: Training tensor contains NaN or Inf values after conversion")
        X_train_tensor = torch.nan_to_num(X_train_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    if torch.isnan(X_test_tensor).any() or torch.isinf(X_test_tensor).any():
        print("Warning: Test tensor contains NaN or Inf values after conversion")
        X_test_tensor = torch.nan_to_num(X_test_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)  # input = target for autoencoders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    print(f"Training autoencoder on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Check for NaN loss
            if torch.isnan(loss).any():
                print(f"NaN loss detected at epoch {epoch+1}. Skipping batch.")
                continue
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average loss for this epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss:.6f}")
    
    # Calculate reconstruction error on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, X_test_tensor)
        print(f"Test loss: {test_loss:.6f}")
        
        # Calculate reconstruction error for each sample
        reconstruction_errors = torch.mean((X_test_tensor - test_outputs)**2, dim=1).cpu().numpy()
    
    return model, reconstruction_errors, y_test, train_losses

# Evaluate the anomaly detection
def evaluate_anomalies(reconstruction_errors, y_true=None, percentile=THRESHOLD_PERCENTILE):
    # Determine threshold based on percentile of errors
    threshold = np.percentile(reconstruction_errors, percentile)
    print(f"Anomaly threshold (at {percentile}th percentile): {threshold:.6f}")
    
    # Predict anomalies
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    # If we have ground truth labels
    if y_true is not None and np.sum(y_true) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        # If possible, calculate ROC AUC using reconstruction errors as scores
        try:
            auc = roc_auc_score(y_true, reconstruction_errors)
        except:
            auc = float('nan')
            
        print(f"Anomaly Detection Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        return {
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'accuracy': accuracy,
            'auc': auc,
            'threshold': threshold,
            'y_pred': y_pred
        }
    else:
        # No ground truth, just report number of anomalies found
        anomaly_count = np.sum(y_pred)
        anomaly_percent = (anomaly_count / len(y_pred)) * 100
        print(f"Detected {anomaly_count} anomalies ({anomaly_percent:.2f}% of test data)")
        
        return {
            'threshold': threshold,
            'anomaly_count': anomaly_count,
            'anomaly_percent': anomaly_percent,
            'y_pred': y_pred
        }

# Visualize results
def visualize_results(reconstruction_errors, y_test=None, threshold=None, results=None):
    plt.figure(figsize=(12, 6))
    
    # Plot reconstruction error distribution
    plt.subplot(1, 2, 1)
    plt.hist(reconstruction_errors, bins=50, alpha=0.7)
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    
    # Plot training loss
    if results and 'train_losses' in results:
        plt.subplot(1, 2, 2)
        plt.plot(results['train_losses'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig('autoencoder_results.png')
    print("Results visualization saved to 'autoencoder_results.png'")

def main():
    # Load and preprocess data
    X, y, scaler = load_preprocess_data()
    
    # Train autoencoder
    model, reconstruction_errors, y_test, train_losses = train_autoencoder(X, y)
    
    # Evaluate results
    results = evaluate_anomalies(reconstruction_errors, y_test)
    results['train_losses'] = train_losses
    
    # Visualize
    visualize_results(reconstruction_errors, y_test, results['threshold'], results)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': results['threshold'],
        'scaler': scaler
    }, 'autoencoder_model.pth')
    print("Model saved to 'autoencoder_model.pth'")

if __name__ == "__main__":
    main() 