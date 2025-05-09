#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from datetime import datetime

def load_data(csv_file):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    return df

def add_anomaly_labels(df):
    """
    Add 'is_anomaly' column based on is_error and has_notanf columns
    
    An anomaly is defined as:
    - is_error is True OR has_notanf is True
    """
    # Create default columns if they don't exist
    if 'is_error' not in df.columns:
        print("Warning: 'is_error' column not found. Creating with default value False.")
        df['is_error'] = False
    
    if 'has_notanf' not in df.columns:
        print("Warning: 'has_notanf' column not found. Creating with default value False.")
        df['has_notanf'] = False
    
    # Create the is_anomaly column based on conditions
    df['is_anomaly'] = (df['is_error'] | df['has_notanf'])
    
    # Count anomalies
    anomaly_count = df['is_anomaly'].sum()
    total_count = len(df)
    anomaly_percentage = (anomaly_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Identified {anomaly_count} anomalies out of {total_count} records ({anomaly_percentage:.2f}%)")
    print(f"Anomaly distribution by cause:")
    print(f"  - Error messages (is_error=True): {df['is_error'].sum()} records")
    print(f"  - has_notanf=True: {df['has_notanf'].sum()} records")
    print(f"  - Both conditions: {(df['is_error'] & df['has_notanf']).sum()} records")
    
    return df

def save_labeled_data(df, input_file, output_file=None):
    """Save the dataframe with anomaly labels to a CSV file"""
    if output_file is None:
        # Generate default output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_labeled_{timestamp}{ext}"
    
    print(f"Saving labeled data to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} rows to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Add anomaly labels to BERT embeddings based on is_error and has_notanf')
    parser.add_argument('--input', help='Path to CSV file with BERT embeddings')
    parser.add_argument('-o', '--output_file', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Load the data
    df = load_data(args.input_file)
    
    # Add anomaly labels
    df_labeled = add_anomaly_labels(df)
    
    # Save the labeled data
    save_labeled_data(df_labeled, args.input_file, args.output_file)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 