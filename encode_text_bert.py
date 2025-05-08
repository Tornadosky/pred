#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def load_data(csv_file, data_column='data'):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # Drop rows where data_column is NaN (optional)
    if df[data_column].isna().any():
        initial_count = len(df)
        df = df.dropna(subset=[data_column])
        print(f"Dropped {initial_count - len(df)} rows with NaN values in '{data_column}' column")
    
    return df

def get_bert_embedding(text, tokenizer, model):
    """Get BERT [CLS] token embedding for a single text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    return cls_embedding.squeeze().numpy()

def process_embeddings(df, data_column, batch_size=32):
    """Process the dataframe and compute embeddings for the data column"""
    # Load BERT tokenizer and model
    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()  # Ensure model is in inference mode
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Compute embeddings
    print(f"Computing embeddings for {len(df)} texts...")
    embeddings = []
    
    # Process in batches for progress display
    for i in tqdm(range(len(df)), desc="Computing BERT embeddings"):
        text = df.iloc[i][data_column]
        embedding = get_bert_embedding(text, tokenizer, model)
        embeddings.append(embedding.tolist())
    
    # Add embeddings to dataframe
    df_with_embeddings = df.copy()
    df_with_embeddings[f"{data_column}_embedding"] = embeddings
    
    # Remove the original text column if desired
    # df_with_embeddings = df_with_embeddings.drop(columns=[data_column])
    
    return df_with_embeddings

def main():
    parser = argparse.ArgumentParser(description='Encode text with BERT and save to CSV')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', 
                        help='Name of the column containing text data (default: data)')
    parser.add_argument('--output_dir', default='.', 
                        help='Directory to save the output CSV file (default: current directory)')
    parser.add_argument('--drop_text', action='store_true',
                        help='Remove the original text column from the output CSV')
    
    args = parser.parse_args()
    
    # Load the CSV data
    df = load_data(args.csv_file, data_column=args.data_column)
    
    # Check if the specified data column exists
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Process the dataframe and compute embeddings
    df_with_embeddings = process_embeddings(df, args.data_column)
    
    # Drop original text column if specified
    if args.drop_text:
        df_with_embeddings = df_with_embeddings.drop(columns=[args.data_column])
        print(f"Dropped original '{args.data_column}' column from output")
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_filename}_bert_embeddings_{timestamp}.csv")
    
    # Save to CSV
    print(f"Saving embeddings to {output_file}...")
    df_with_embeddings.to_csv(output_file, index=False)
    
    # Print summary
    embedding_col = f"{args.data_column}_embedding"
    embedding_size = len(df_with_embeddings[embedding_col].iloc[0])
    print(f"Saved {len(df_with_embeddings)} rows to {output_file}")
    print(f"Embedding dimensions: {embedding_size}")
    print(f"Process completed successfully!")

if __name__ == "__main__":
    main() 