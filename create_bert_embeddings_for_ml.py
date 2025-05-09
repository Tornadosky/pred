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

def load_data(csv_file):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    return df

def get_bert_embedding(text, tokenizer, model, device):
    """Get BERT [CLS] token embedding for a single text"""
    # Handle empty or NaN text
    if pd.isna(text) or text == "":
        # Return zeros with correct shape (assuming bert-base-uncased with 768 dimensions)
        return np.zeros(768)
    
    # Create inputs and move to the correct device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    
    # Move the result back to CPU for numpy conversion
    return cls_embedding.cpu().squeeze().numpy()

def process_embeddings(df, text_column='message'):
    """Process each row in the dataframe and compute embeddings for the text column"""
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BERT tokenizer and model
    print("Loading BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()  # Ensure model is in inference mode
    
    # Move model to device
    model = model.to(device)
    
    # Create a copy of the dataframe
    df_with_embeddings = df.copy()
    
    # Compute embeddings for each row
    print(f"Computing embeddings for {len(df)} rows...")
    
    # Method 1: Store embeddings as JSON strings
    embeddings_json = []
    
    # Method 2: Create separate columns for each dimension
    embedding_dim = 768  # BERT base hidden size
    embedding_columns = {}
    
    for i in tqdm(range(len(df)), desc="Computing BERT embeddings"):
        # Get the text from the message column for this row
        text = str(df[text_column].iloc[i])
        # Get embedding for this text
        embedding = get_bert_embedding(text, tokenizer, model, device)
        
        # Method 1: Store as JSON
        embeddings_json.append(json.dumps(embedding.tolist()))
        
        # Method 2: Store in separate columns
        for j in range(embedding_dim):
            col_name = f"bert_dim_{j}"
            if col_name not in embedding_columns:
                embedding_columns[col_name] = []
            embedding_columns[col_name].append(embedding[j])
    
    # Add JSON embeddings as a column
    df_with_embeddings['bert_embeddings'] = embeddings_json
    
    # Add individual dimension columns
    for col_name, values in embedding_columns.items():
        df_with_embeddings[col_name] = values
    
    # Print sample of the embeddings
    sample_embedding = json.loads(embeddings_json[0])
    print(f"\nSample embedding dimensions: {len(sample_embedding)}")
    
    return df_with_embeddings

def save_embeddings(df, output_dir='.', base_filename=None, include_dimensions=True):
    """Save embeddings to various formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if base_filename is None:
        base_filename = "data_with_bert_embeddings"
    
    # Save complete dataframe with JSON embeddings
    output_file = os.path.join(output_dir, f"{base_filename}_{timestamp}.csv")
    print(f"\nSaving dataframe with JSON embeddings to {output_file}...")
    df.to_csv(output_file, index=False)
    
    if include_dimensions:
        # Create ML-ready version with just the individual dimension columns
        ml_columns = [col for col in df.columns if col.startswith('bert_dim_')]
        
        # Add any non-embedding columns that might be useful for modeling
        # (excluding the JSON embeddings column itself)
        for col in df.columns:
            if col != 'bert_embeddings' and not col.startswith('bert_dim_'):
                ml_columns.append(col)
        
        ml_df = df[ml_columns]
        ml_output_file = os.path.join(output_dir, f"{base_filename}_ml_ready_{timestamp}.csv")
        print(f"Saving ML-ready dataframe to {ml_output_file}...")
        ml_df.to_csv(ml_output_file, index=False)
        
        print(f"ML-ready dataframe shape: {ml_df.shape}")
    
    print(f"Saved {len(df)} rows successfully")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Create BERT embeddings from message column for ML models')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--text_column', default='message', 
                        help='Name of the column containing text to embed (default: message)')
    parser.add_argument('--output_dir', default='.', 
                        help='Directory to save the output CSV files (default: current directory)')
    parser.add_argument('--no_dimensions', action='store_true',
                        help='Do not create individual dimension columns (saves space but less ML-friendly)')
    
    args = parser.parse_args()
    
    # Load the CSV data
    df = load_data(args.csv_file)
    
    # Check if the specified text column exists
    if args.text_column not in df.columns:
        print(f"Error: Column '{args.text_column}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Process the dataframe and compute embeddings row by row
    df_with_embeddings = process_embeddings(df, text_column=args.text_column)
    
    # Create output filename base from input file
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    
    # Save to CSV files
    save_embeddings(df_with_embeddings, 
                   output_dir=args.output_dir, 
                   base_filename=base_filename, 
                   include_dimensions=not args.no_dimensions)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 