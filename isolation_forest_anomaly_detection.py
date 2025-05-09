import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, accuracy_score

# Configuration
RANDOM_STATE = 42
N_ESTIMATORS = 100
CONTAMINATION = 'auto'
MAX_SAMPLES = 'auto'  # or specify a number
MODEL_FILE = 'isolation_forest_model.pkl'

def load_preprocess_data(file_path, remove_date_time=True):
    """Load and preprocess data for Isolation Forest"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Store original data
    original_df = df.copy()
    
    # Check if the data has labels
    has_labels = 'is_anomaly' in df.columns
    
    if has_labels:
        # Extract labels
        labels = df['is_anomaly'].values
        print(f"Found {sum(labels)} labeled anomalies out of {len(labels)} samples")
        # Remove label column for training
        df = df.drop('is_anomaly', axis=1)
    else:
        labels = None
        print("No labeled data found. Will perform unsupervised anomaly detection.")
    
    # Remove date and time if requested
    if remove_date_time and 'date' in df.columns and 'time' in df.columns:
        ref_columns = ['date', 'time']
        ref_df = df[ref_columns].copy()
        df = df.drop(ref_columns, axis=1)
    elif 'date' in df.columns:
        ref_columns = ['date']
        ref_df = df[ref_columns].copy()
        df = df.drop(ref_columns, axis=1)
    elif 'time' in df.columns:
        ref_columns = ['time']
        ref_df = df[ref_columns].copy()
        df = df.drop(ref_columns, axis=1)
    else:
        ref_df = pd.DataFrame(index=df.index)
    
    # Handle missing values - fill NaN values
    for col in df.columns:
        if df[col].dtype.kind in 'fc':  # float or complex
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype.kind == 'i':  # integer
            df[col] = df[col].fillna(0)
        else:  # object or categorical
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "UNKNOWN")
    
    # Convert categorical columns to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    # Handle any remaining NaN or inf values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Preprocessed data shape: {X_scaled.shape}")
    return X_scaled, labels, ref_df, original_df, scaler

def train_isolation_forest(X, contamination=CONTAMINATION, n_estimators=N_ESTIMATORS, 
                           max_samples=MAX_SAMPLES, random_state=RANDOM_STATE):
    """Train an Isolation Forest model"""
    print(f"Training Isolation Forest with {n_estimators} trees...")
    
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    model.fit(X)
    print("Model training complete.")
    
    return model

def evaluate_model(model, X, true_labels=None, ref_df=None, original_df=None):
    """Evaluate the Isolation Forest model"""
    # Get anomaly scores (more negative = more anomalous)
    raw_scores = model.score_samples(X)
    
    # Convert to a more intuitive form (higher = more anomalous)
    anomaly_scores = -raw_scores  # Negate so higher score = more anomalous
    
    # Get binary predictions (-1 for outliers, 1 for inliers)
    raw_predictions = model.predict(X)
    
    # Convert to 0 for normal, 1 for anomaly
    predicted_labels = np.where(raw_predictions == -1, 1, 0)
    
    # Calculate anomaly ratio
    anomaly_count = np.sum(predicted_labels)
    total_count = len(predicted_labels)
    anomaly_ratio = anomaly_count / total_count
    
    print(f"Detected {anomaly_count} anomalies out of {total_count} samples ({anomaly_ratio:.2%})")
    
    results = {
        'anomaly_scores': anomaly_scores,
        'predicted_labels': predicted_labels
    }
    
    # If we have true labels, calculate performance metrics
    if true_labels is not None:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='binary', zero_division=0
        )
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        try:
            auc_score = roc_auc_score(true_labels, anomaly_scores)
        except:
            auc_score = float('nan')
        
        print(f"Model Performance:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        
        # Find the best threshold using F1 score
        precision_curve, recall_curve, thresholds = precision_recall_curve(true_labels, anomaly_scores)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        best_f1 = f1_scores[best_idx]
        
        # Get predictions using best threshold
        best_predicted = (anomaly_scores > best_threshold).astype(int)
        best_precision = precision_curve[best_idx]
        best_recall = recall_curve[best_idx]
        best_accuracy = accuracy_score(true_labels, best_predicted)
        
        print("\nBest threshold for F1 score:")
        print(f"  Threshold: {best_threshold:.6f}")
        print(f"  Precision: {best_precision:.4f}")
        print(f"  Recall: {best_recall:.4f}")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Accuracy: {best_accuracy:.4f}")
        
        results.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc_score,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_predicted': best_predicted
        })
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, best_predicted)
        
        # Visualize results
        visualize_results(anomaly_scores, true_labels, results)
    
    # Combine results into a DataFrame
    result_df = pd.DataFrame({
        'Anomaly_Score': anomaly_scores,
        'Predicted_Label': predicted_labels
    })
    
    # Add reference columns if available
    if ref_df is not None and not ref_df.empty:
        result_df = pd.concat([result_df, ref_df.reset_index(drop=True)], axis=1)
    
    # Add original true labels if available
    if true_labels is not None:
        result_df['True_Label'] = true_labels
    
    # Export top anomalies
    top_indices = np.argsort(anomaly_scores)[-100:][::-1]  # Top 100 anomalies
    top_anomalies = result_df.iloc[top_indices].copy()
    
    # Add additional columns from original data for top anomalies if available
    if original_df is not None:
        useful_cols = []
        for col in ['operation_type', 'message_length', 'is_error', 'is_warning', 'has_notanf']:
            if col in original_df.columns:
                useful_cols.append(col)
        
        if useful_cols:
            for col in useful_cols:
                top_anomalies[col] = original_df[col].iloc[top_indices].values
    
    top_anomalies = top_anomalies.sort_values('Anomaly_Score', ascending=False).reset_index(drop=True)
    
    # Save results
    result_df.to_csv('isolation_forest_all_results.csv', index=False)
    top_anomalies.to_csv('isolation_forest_top_anomalies.csv', index=False)
    
    print("\nResults saved to 'isolation_forest_all_results.csv'")
    print("Top anomalies saved to 'isolation_forest_top_anomalies.csv'")
    
    return results

def visualize_results(anomaly_scores, true_labels, results, output_prefix="isolation_forest"):
    """Create visualizations for the model evaluation"""
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Score distribution by true label
    plt.subplot(2, 2, 1)
    sns.histplot(
        x=anomaly_scores, 
        hue=true_labels, 
        bins=50, 
        kde=True,
        element="step",
        common_norm=False
    )
    
    if 'best_threshold' in results:
        plt.axvline(
            x=results['best_threshold'], 
            color='r', 
            linestyle='--', 
            label=f'Best F1 Threshold: {results["best_threshold"]:.6f}'
        )
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Distribution of Anomaly Scores by True Label')
    plt.legend(['Best F1 Threshold', 'Normal', 'Anomaly'])
    
    # Plot 2: ROC curve
    plt.subplot(2, 2, 2)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {results["auc"]:.4f})')
    
    # Plot 3: Precision-Recall curve
    plt.subplot(2, 2, 3)
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, anomaly_scores)
    plt.plot(recall_curve, precision_curve)
    
    if 'best_precision' in results and 'best_recall' in results:
        plt.scatter(
            results['best_recall'], 
            results['best_precision'], 
            color='red', 
            s=100, 
            label=f'Best F1: {results["best_f1"]:.4f}'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Plot 4: Confusion Matrix
    plt.subplot(2, 2, 4)
    if 'best_predicted' in results:
        cm = confusion_matrix(true_labels, results['best_predicted'])
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
    else:
        cm = confusion_matrix(true_labels, results['predicted_labels'])
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
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plots.png")
    print(f"Visualization saved to {output_prefix}_plots.png")

def save_model(model, scaler, filename=MODEL_FILE):
    """Save the trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler
    }
    
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate Isolation Forest for anomaly detection')
    parser.add_argument('--data', required=True, help='Path to the data CSV file')
    parser.add_argument('--n_estimators', type=int, default=N_ESTIMATORS, help='Number of trees in the forest')
    parser.add_argument('--contamination', default=CONTAMINATION, help='Expected proportion of anomalies')
    parser.add_argument('--max_samples', default=MAX_SAMPLES, help='Number of samples to train each base estimator')
    parser.add_argument('--output_model', default=MODEL_FILE, help='Path to save the model')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    X, labels, ref_df, original_df, scaler = load_preprocess_data(args.data)
    
    # Parse contamination
    contamination = args.contamination
    if contamination != 'auto' and not isinstance(contamination, float):
        try:
            contamination = float(contamination)
        except:
            print(f"Warning: Could not convert contamination '{contamination}' to float. Using 'auto' instead.")
            contamination = 'auto'
    
    # Parse max_samples
    max_samples = args.max_samples
    if max_samples != 'auto' and not isinstance(max_samples, int):
        try:
            max_samples = int(max_samples)
        except:
            print(f"Warning: Could not convert max_samples '{max_samples}' to int. Using 'auto' instead.")
            max_samples = 'auto'
    
    # Train the model
    model = train_isolation_forest(
        X, 
        contamination=contamination,
        n_estimators=args.n_estimators,
        max_samples=max_samples
    )
    
    # Evaluate the model
    evaluate_model(model, X, labels, ref_df, original_df)
    
    # Save the model
    save_model(model, scaler, args.output_model)

if __name__ == "__main__":
    main() 