import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from autoencoder_anomaly_detection import Autoencoder, load_preprocess_data

def load_labeled_data(labeled_file):
    """Load a CSV file with anomaly labels"""
    print(f"Loading labeled data from {labeled_file}...")
    df = pd.read_csv(labeled_file)
    
    if 'is_anomaly' not in df.columns:
        raise ValueError(f"The file {labeled_file} does not contain an 'is_anomaly' column")
    
    print(f"Loaded {len(df)} records, with {df['is_anomaly'].sum()} labeled anomalies")
    return df

def load_autoencoder_model(model_path):
    """Load the trained autoencoder model"""
    print(f"Loading autoencoder model from {model_path}...")
    
    try:
        # Try loading with newer PyTorch safe globals
        import torch.serialization
        import numpy
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except:
        # Fall back to loading with weights_only=False
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    return checkpoint

def calculate_reconstruction_errors(model, X):
    """Calculate reconstruction errors for all samples"""
    print("Calculating reconstruction errors...")
    X_tensor = torch.FloatTensor(X)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Calculate reconstruction error
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
    
    return errors

def evaluate_performance(errors, true_labels, threshold=None, percentiles=[90, 95, 99]):
    """Evaluate the performance of anomaly detection at different thresholds"""
    results = {}
    
    if threshold is None:
        # Try different percentile thresholds
        for p in percentiles:
            threshold_p = np.percentile(errors, p)
            predicted_p = (errors > threshold_p).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predicted_p, average='binary', zero_division=0
            )
            
            try:
                auc_score = roc_auc_score(true_labels, errors)
            except:
                auc_score = float('nan')
                
            accuracy = accuracy_score(true_labels, predicted_p)
            
            results[f'percentile_{p}'] = {
                'threshold': threshold_p,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'auc': auc_score,
                'predicted': predicted_p
            }
            
            print(f"Performance at {p}th percentile threshold ({threshold_p:.6f}):")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc_score:.4f}")
    else:
        # Use the provided threshold
        predicted = (errors > threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted, average='binary', zero_division=0
        )
        
        try:
            auc_score = roc_auc_score(true_labels, errors)
        except:
            auc_score = float('nan')
            
        accuracy = accuracy_score(true_labels, predicted)
        
        results['custom_threshold'] = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc_score,
            'predicted': predicted
        }
        
        print(f"Performance at custom threshold ({threshold:.6f}):")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc_score:.4f}")
    
    # Find best F1 score threshold
    precision_curve, recall_curve, thresholds = precision_recall_curve(true_labels, errors)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]
    
    # Get predictions using best threshold
    best_predicted = (errors > best_threshold).astype(int)
    best_precision = precision_curve[best_idx]
    best_recall = recall_curve[best_idx]
    best_accuracy = accuracy_score(true_labels, best_predicted)
    
    results['best_f1'] = {
        'threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1,
        'accuracy': best_accuracy,
        'auc': auc_score,
        'predicted': best_predicted
    }
    
    print("\nBest threshold for F1 score:")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall: {best_recall:.4f}")
    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    
    return results

def visualize_results(errors, true_labels, results, output_prefix="evaluation"):
    """Visualize the evaluation results"""
    # Create confusion matrix for best F1 threshold
    best_predicted = results['best_f1']['predicted']
    cm = confusion_matrix(true_labels, best_predicted)
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Error distribution by true label
    plt.subplot(2, 2, 1)
    sns.histplot(
        x=errors, 
        hue=true_labels, 
        bins=50, 
        kde=True,
        element="step",
        common_norm=False
    )
    plt.axvline(
        x=results['best_f1']['threshold'], 
        color='r', 
        linestyle='--', 
        label=f'Best F1 Threshold: {results["best_f1"]["threshold"]:.6f}'
    )
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Reconstruction Errors by True Label')
    plt.legend(['Best F1 Threshold', 'Normal', 'Anomaly'])
    
    # Plot 2: ROC curve
    plt.subplot(2, 2, 2)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(true_labels, errors)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {results["best_f1"]["auc"]:.4f})')
    
    # Plot 3: Precision-Recall curve
    plt.subplot(2, 2, 3)
    plt.plot(recall_curve, precision_curve)
    plt.scatter(
        results['best_f1']['recall'], 
        results['best_f1']['precision'], 
        color='red', 
        s=100, 
        label=f'Best F1: {results["best_f1"]["f1"]:.4f}'
    )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Plot 4: Confusion Matrix
    plt.subplot(2, 2, 4)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly']
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Best F1 Threshold)')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plots.png")
    print(f"Visualization saved to {output_prefix}_plots.png")
    
    # Save detailed results to CSV
    df_results = pd.DataFrame({
        'true_label': true_labels,
        'reconstruction_error': errors,
        'predicted_best_f1': best_predicted
    })
    df_results.to_csv(f"{output_prefix}_detailed_results.csv", index=False)
    print(f"Detailed results saved to {output_prefix}_detailed_results.csv")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate autoencoder against labeled anomalies')
    parser.add_argument('--model', default='autoencoder_model.pth', help='Path to the autoencoder model')
    parser.add_argument('--labeled_data', required=True, help='Path to labeled data CSV file')
    parser.add_argument('--output_prefix', default='autoencoder_evaluation', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Load labeled data
    labeled_df = load_labeled_data(args.labeled_data)
    true_labels = labeled_df['is_anomaly'].values
    
    # Load autoencoder model
    checkpoint = load_autoencoder_model(args.model)
    
    # Load and preprocess data (same as during training)
    X_scaled, _, scaler = load_preprocess_data(args.labeled_data)
    
    # Create model with correct input dimensions
    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate reconstruction errors
    reconstruction_errors = calculate_reconstruction_errors(model, X_scaled)
    
    # Evaluate performance
    # Use the threshold from training if available
    threshold = checkpoint.get('threshold', None)
    results = evaluate_performance(reconstruction_errors, true_labels, threshold)
    
    # Visualize results
    visualize_results(reconstruction_errors, true_labels, results, args.output_prefix)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 