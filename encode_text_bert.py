#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_data(csv_file):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    return df

def encode_texts_with_bert(texts, model_name='bert-base-uncased'):
    """Encode texts using SentenceTransformer with BERT model"""
    print(f"Loading SentenceTransformer with model: {model_name}")
    bert_model = SentenceTransformer(model_name)
    
    print("Encoding texts...")
    # Use tqdm to show progress
    encodings = []
    for text in tqdm(texts, desc="Encoding texts"):
        encodings.append(bert_model.encode(text))
    
    return encodings

def save_to_csv(df, encodings, output_file):
    """Save dataframe with encodings to a new CSV file"""
    print(f"Preparing to save encodings to {output_file}...")
    
    # Create a copy of the original dataframe
    df_with_encodings = df.copy()
    
    # Convert encodings to string to store in CSV
    encoding_strings = [json.dumps(encoding.tolist()) for encoding in encodings]
    df_with_encodings['encoding'] = encoding_strings
    
    # Save to CSV
    df_with_encodings.to_csv(output_file, index=False)
    
    print(f"Saved {len(df_with_encodings)} rows with encodings to {output_file}")
    print(f"Encoding dimensions: {len(json.loads(encoding_strings[0]))}")
    
    # Calculate approximate size
    total_size_mb = sum(len(enc) for enc in encoding_strings) * 8 / 1_000_000
    print(f"Total size of encodings: {total_size_mb:.2f} MB (approximate)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Encode text with BERT and save to CSV')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', 
                        help='Name of the column containing text data (default: data)')
    parser.add_argument('--output_dir', default='.', 
                        help='Directory to save the output CSV file (default: current directory)')
    parser.add_argument('--model_name', default='bert-base-uncased',
                        help='SentenceTransformer model to use (default: bert-base-uncased)')
    
    args = parser.parse_args()
    
    # Load the CSV data
    df = load_data(args.csv_file)
    
    # Check if the specified data column exists
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Get texts from the data column
    texts = df[args.data_column].fillna('').astype(str).tolist()
    print(f"Processing {len(texts)} text entries")
    
    # Encode the texts using SentenceTransformer with BERT
    encodings = encode_texts_with_bert(texts, model_name=args.model_name)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_filename}_bert_encoded_{timestamp}.csv")
    
    # Save the dataframe with encodings to a new CSV
    save_to_csv(df, encodings, output_file)
    print(f"\nProcess completed successfully!")

if __name__ == "__main__":
    main() 