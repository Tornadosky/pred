#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
from datetime import datetime
import torch
import json
import sys

def load_data(csv_file):
    """Load data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    return df

def get_sentence_transformer_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings using sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Using sentence-transformers model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except ImportError:
        print("sentence-transformers is not installed. Trying to install it...")
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
        print(f"Using sentence-transformers model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings

def get_bert_embeddings(texts, batch_size=32, model_name='bert-base-uncased'):
    """Get BERT embeddings for a list of texts"""
    from transformers import BertTokenizer, BertModel
    
    print(f"Loading BERT model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize texts
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=128, return_tensors='pt')
        
        # Move to device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Get model output
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Use CLS token embedding as sentence embedding
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings

def save_embeddings_to_csv(df, embeddings, output_file, data_column):
    """Save the original dataframe with embeddings to a CSV file"""
    print(f"Saving embeddings to {output_file}...")
    
    # Create a copy of the original dataframe
    df_with_embeddings = df.copy()
    
    # Convert embeddings to string to store in CSV
    embeddings_list = [json.dumps(embedding.tolist()) for embedding in embeddings]
    df_with_embeddings['embedding'] = embeddings_list
    
    # Save to CSV
    df_with_embeddings.to_csv(output_file, index=False)
    print(f"Saved {len(df_with_embeddings)} rows with embeddings to {output_file}")
    
    # Output some stats
    embedding_sizes = [len(json.loads(emb)) for emb in embeddings_list]
    print(f"Embedding dimensions: {embedding_sizes[0]}")
    print(f"Total size of embeddings: {sum(embedding_sizes) * 8 / 1_000_000:.2f} MB (approximate)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Create embeddings from text data in a CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', help='Name of the column containing text data')
    parser.add_argument('--output_dir', default='.', help='Directory to save the output CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--embedding_type', choices=['sentence-transformer', 'bert'], 
                       default='sentence-transformer', help='Type of embedding to use')
    parser.add_argument('--model_name', default='all-MiniLM-L6-v2', 
                       help='Model name (default: all-MiniLM-L6-v2 for sentence-transformer, bert-base-uncased for BERT)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.csv_file)
    
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        return
    
    # Get texts from data column
    texts = df[args.data_column].fillna('').astype(str).tolist()
    print(f"Processing {len(texts)} text entries")
    
    # Generate embeddings
    if args.embedding_type == 'sentence-transformer':
        embeddings = get_sentence_transformer_embeddings(texts, model_name=args.model_name)
    else:  # bert
        model_name = args.model_name if args.model_name != 'all-MiniLM-L6-v2' else 'bert-base-uncased'
        embeddings = get_bert_embeddings(texts, batch_size=args.batch_size, model_name=model_name)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_filename}_embeddings_{timestamp}.csv")
    
    # Save embeddings
    save_embeddings_to_csv(df, embeddings, output_file, args.data_column)

if __name__ == "__main__":
    main() 