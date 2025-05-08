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

def process_embeddings(df, data_column='data'):
    """Process each row in the dataframe and compute embeddings for the data column"""
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
    embeddings = []
    
    for i in tqdm(range(len(df)), desc="Computing BERT embeddings"):
        # Get the text from the data column for this row
        text = str(df[data_column].iloc[i])
        # Get embedding for this text
        embedding = get_bert_embedding(text, tokenizer, model, device)
        # Add to list as JSON serializable format
        embeddings.append(json.dumps(embedding.tolist()))
    
    # Add embeddings as a new column
    df_with_embeddings['bert_embeddings'] = embeddings
    
    # Print sample of the embeddings
    sample_embedding = json.loads(embeddings[0])
    print(f"\nSample embedding dimensions: {len(sample_embedding)}")
    
    return df_with_embeddings

def main():
    parser = argparse.ArgumentParser(description='Encode text with BERT and save to CSV')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', 
                        help='Name of the column containing text data (default: data)')
    parser.add_argument('--output_dir', default='.', 
                        help='Directory to save the output CSV file (default: current directory)')
    
    args = parser.parse_args()
    
    # Load the CSV data
    df = load_data(args.csv_file)
    
    # Check if the specified data column exists
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Process the dataframe and compute embeddings row by row
    df_with_embeddings = process_embeddings(df, data_column=args.data_column)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_filename}_with_embeddings_{timestamp}.csv")
    
    # Save to CSV
    print(f"\nSaving dataframe with embeddings to {output_file}...")
    df_with_embeddings.to_csv(output_file, index=False)
    
    print(f"Saved {len(df_with_embeddings)} rows to {output_file}")
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 