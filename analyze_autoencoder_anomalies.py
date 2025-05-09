import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from autoencoder_anomaly_detection import Autoencoder

# Set paths and configuration
MODEL_PATH = 'autoencoder_model.pth'
DATA_PATH = 'data_with_word2vec_embeddings_expanded.csv'
TOP_N_ANOMALIES = 100

def load_model_and_data():
    """Load the trained autoencoder model and the original dataset"""
    print(f"Loading model from {MODEL_PATH}...")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Run 'autoencoder_anomaly_detection.py' first.")
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Get the scaler from checkpoint
    scaler = checkpoint.get('scaler')
    
    # Get anomaly threshold from checkpoint
    threshold = checkpoint.get('threshold', None)
    
    # Load and preprocess data (similar to training script but maintaining row identity)
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Store original df before preprocessing to reference later
    original_df = df.copy()
    
    # Remove date and time but keep as reference columns
    ref_columns = ['date', 'time']
    ref_df = df[ref_columns].copy()
    df = df.drop(ref_columns, axis=1)
    
    # Check if the data has labels
    has_labels = 'is_anomaly' in df.columns
    if has_labels:
        y = df['is_anomaly'].values
        X = df.drop('is_anomaly', axis=1)
    else:
        y = None
        X = df
    
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
    
    # Use the same scaler from training to transform the data
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        # If no scaler in checkpoint, create a new one
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Handle any remaining NaN values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Load model architecture
    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim=input_dim)
    
    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, X_scaled, original_df, ref_df, threshold, y

def identify_top_anomalies(model, X_scaled, n=TOP_N_ANOMALIES):
    """Calculate reconstruction errors and identify top N anomalies"""
    print("Calculating reconstruction errors...")
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Calculate reconstruction error for each sample
    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Calculate per-sample reconstruction error (mean squared error across features)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    
    # Get indices of top N errors
    top_indices = np.argsort(errors)[-n:][::-1]  # Sort in descending order
    top_errors = errors[top_indices]
    
    print(f"Top {n} reconstruction errors: {top_errors[:5]}...")
    
    return top_indices, top_errors

def visualize_anomalies(original_df, ref_df, top_indices, errors, threshold=None):
    """Visualize the top anomalies"""
    # Create a dataframe of the top anomalies with reference columns and error scores
    anomaly_df = pd.DataFrame({
        'Index': top_indices,
        'Reconstruction_Error': errors
    })
    
    # Add reference columns
    anomaly_df = pd.concat([
        anomaly_df, 
        ref_df.iloc[top_indices].reset_index(drop=True)
    ], axis=1)
    
    # Add a few columns from original data that might be useful for analysis
    # Choose columns that might be relevant for anomaly interpretation
    useful_cols = []
    if 'operation_type' in original_df.columns:
        useful_cols.append('operation_type')
    if 'message_length' in original_df.columns:
        useful_cols.append('message_length')
    if 'is_error' in original_df.columns:
        useful_cols.append('is_error')
    if 'is_warning' in original_df.columns:
        useful_cols.append('is_warning')
    
    for col in useful_cols:
        anomaly_df[col] = original_df[col].iloc[top_indices].values
    
    # Print the top anomalies
    print("\nTop Anomalies:")
    pd.set_option('display.max_columns', 10)
    print(anomaly_df.head(10))
    
    # Save to CSV
    output_file = 'top_anomalies.csv'
    anomaly_df.to_csv(output_file, index=False)
    print(f"\nSaved all {len(anomaly_df)} anomalies to {output_file}")
    
    # Visualize reconstruction error distribution
    plt.figure(figsize=(12, 6))
    
    # Histogram of all errors
    plt.subplot(1, 2, 1)
    sns.histplot(errors, bins=50, kde=True)
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors for Top Anomalies')
    plt.legend()
    
    # Scatter plot of top errors by index (to see if anomalies cluster in time)
    plt.subplot(1, 2, 2)
    plt.scatter(top_indices, errors, alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Top Anomalies by Index')
    
    plt.tight_layout()
    plt.savefig('top_anomalies_plot.png')
    print("Visualization saved to 'top_anomalies_plot.png'")

def main():
    # Load the model and data
    model, X_scaled, original_df, ref_df, threshold, y = load_model_and_data()
    
    # Get top anomalies
    top_indices, top_errors = identify_top_anomalies(model, X_scaled)
    
    # Visualize the results
    visualize_anomalies(original_df, ref_df, top_indices, top_errors, threshold)
    
    # If ground truth labels are available, evaluate detection accuracy on top anomalies
    if y is not None:
        top_true_labels = y[top_indices]
        true_anomaly_count = np.sum(top_true_labels)
        if true_anomaly_count > 0:
            accuracy = (true_anomaly_count / len(top_indices)) * 100
            print(f"\nGround Truth Validation:")
            print(f"True anomalies among top {len(top_indices)}: {true_anomaly_count} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main() 