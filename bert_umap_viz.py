#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch
from umap import UMAP
import argparse
from tqdm import tqdm
import os
import shutil
from datetime import datetime

def load_data(csv_file, preserve_original=True):
    """Load data from CSV file, optionally creating a copy first"""
    print(f"Loading data from {csv_file}...")
    
    # If preserve_original is True, create a copy of the file for processing
    working_file = csv_file
    if preserve_original:
        file_name, file_ext = os.path.splitext(csv_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        working_file = f"{file_name}_working_{timestamp}{file_ext}"
        
        print(f"Creating a copy of the original file: {working_file}")
        shutil.copy2(csv_file, working_file)
    
    # Load the data
    df = pd.read_csv(working_file)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # If we created a working copy, delete it after loading
    if preserve_original and working_file != csv_file:
        os.remove(working_file)
        print(f"Removed temporary working file: {working_file}")
    
    return df

def get_sentence_transformer_embeddings(texts):
    """Alternative embedding method using sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        print("Using sentence-transformers for embeddings")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except ImportError:
        print("sentence-transformers is not installed. Trying to install it...")
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
        print("Using sentence-transformers for embeddings")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings

def get_bert_embeddings(texts, batch_size=32, use_sentence_transformer=False):
    """Get BERT embeddings for a list of texts"""
    # Use sentence-transformers if specified (simpler approach)
    if use_sentence_transformer:
        return get_sentence_transformer_embeddings(texts)
    
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
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
        try:
            # Try the standard way to get embeddings
            if hasattr(output, 'last_hidden_state'):
                batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            # Alternative for newer transformers versions
            elif isinstance(output, tuple) and len(output) > 0:
                batch_embeddings = output[0][:, 0, :].cpu().numpy()
            # Direct dictionary access
            elif isinstance(output, dict) and 'last_hidden_state' in output:
                batch_embeddings = output['last_hidden_state'][:, 0, :].cpu().numpy()
            else:
                print(f"Warning: Unexpected model output format: {type(output)}")
                print("Trying alternative approach...")
                # Fallback for other output structures
                if isinstance(output, dict):
                    for key in output:
                        print(f"Available key: {key}")
                    # Try first tensor in the dict
                    for key, value in output.items():
                        if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                            print(f"Using {key} for embeddings")
                            batch_embeddings = value[:, 0, :].cpu().numpy()
                            break
                    else:
                        raise ValueError("Could not find appropriate tensor in model output")
                else:
                    raise ValueError(f"Unsupported output type: {type(output)}")
        except Exception as e:
            print(f"Error extracting embeddings: {str(e)}")
            print(f"Output type: {type(output)}")
            # Print more debug info
            if isinstance(output, dict):
                print(f"Output keys: {list(output.keys())}")
            elif isinstance(output, tuple):
                print(f"Output tuple length: {len(output)}")
                for i, item in enumerate(output):
                    print(f"Item {i} type: {type(item)}")
            
            print("\nFalling back to sentence-transformers for embeddings...")
            return get_sentence_transformer_embeddings(texts)
        
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings

def reduce_dimensions(embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
    """Reduce dimensions of embeddings using UMAP"""
    print("Reducing dimensions with UMAP...")
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                  n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced dimensions to shape: {reduced_embeddings.shape}")
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings, texts=None, n_samples=100, figsize=(12, 10), output_prefix="bert_umap"):
    """Visualize the reduced embeddings"""
    print("Creating visualization...")
    plt.figure(figsize=figsize)
    
    # Plot all points
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                alpha=0.3, s=10, c='blue')
    
    # Optionally add text annotations for a sample of points
    if texts is not None and n_samples > 0:
        n_samples = min(n_samples, len(reduced_embeddings))
        indices = np.random.choice(len(reduced_embeddings), n_samples, replace=False)
        
        for idx in indices:
            text = texts[idx]
            # Truncate long texts
            if len(text) > 30:
                text = text[:27] + "..."
            plt.annotate(text, (reduced_embeddings[idx, 0], reduced_embeddings[idx, 1]), 
                         fontsize=8, alpha=0.7)
    
    plt.title('UMAP Visualization of BERT Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    
    # Save figure with timestamp to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_prefix}_visualization_{timestamp}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Visualization saved to {output_file}")
    
    # Also save the reduced embeddings for later use
    embeddings_file = f"{output_prefix}_embeddings_{timestamp}.npy"
    np.save(embeddings_file, reduced_embeddings)
    print(f"Embeddings saved to {embeddings_file}")
    
    # Show figure
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Embed text data with BERT and visualize with UMAP')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', help='Name of the data column containing text')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for BERT processing')
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors parameter for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='min_dist parameter for UMAP')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of text samples to show in visualization')
    parser.add_argument('--output_prefix', default='bert_umap', help='Prefix for output files')
    parser.add_argument('--preserve_original', action='store_true', help='Preserve the original file by working on a copy')
    parser.add_argument('--use_sentence_transformer', action='store_true', help='Use sentence-transformers for embeddings (simpler approach)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.csv_file, preserve_original=args.preserve_original)
    
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        return
    
    # Get texts from data column
    texts = df[args.data_column].fillna('').astype(str).tolist()
    
    # Generate embeddings (using either BERT or sentence-transformers)
    embeddings = get_bert_embeddings(texts, batch_size=args.batch_size, use_sentence_transformer=args.use_sentence_transformer)
    
    # Reduce dimensions with UMAP
    reduced_embeddings = reduce_dimensions(embeddings, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    
    # Visualize embeddings
    visualize_embeddings(reduced_embeddings, texts, n_samples=args.n_samples, output_prefix=args.output_prefix)

if __name__ == "__main__":
    import sys
    main() 