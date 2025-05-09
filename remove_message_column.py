#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from datetime import datetime

def remove_message_column(input_file, output_file=None):
    """
    Remove the 'message' column from a CSV file since we have BERT embeddings
    
    Args:
        input_file (str): Path to the input CSV file with BERT embeddings
        output_file (str, optional): Path to save the output CSV file without the message column
    
    Returns:
        str: Path to the output file
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    original_columns = list(df.columns)
    original_rows = len(df)
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # Size in MB
    
    print(f"Loaded {original_rows} rows with {len(original_columns)} columns")
    print(f"Original file size: {original_size:.2f} MB")
    
    # Check if message column exists
    if 'message' in df.columns:
        print("Removing 'message' column...")
        df = df.drop('message', axis=1)
        print(f"Column 'message' removed. Remaining columns: {len(df.columns)}")
    else:
        print("Warning: 'message' column not found in the dataset.")
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_no_message_{timestamp}{ext}"
    
    # Save the modified dataframe
    print(f"Saving data without message column to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Calculate size reduction
    new_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
    size_reduction = original_size - new_size
    size_reduction_percent = (size_reduction / original_size) * 100 if original_size > 0 else 0
    
    print(f"New file size: {new_size:.2f} MB")
    print(f"Size reduction: {size_reduction:.2f} MB ({size_reduction_percent:.1f}%)")
    print(f"Saved {len(df)} rows with {len(df.columns)} columns to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Remove message column from a CSV file with BERT embeddings')
    parser.add_argument('input_file', help='Path to CSV file with BERT embeddings and message column')
    parser.add_argument('-o', '--output_file', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    # Process the file
    remove_message_column(args.input_file, args.output_file)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 