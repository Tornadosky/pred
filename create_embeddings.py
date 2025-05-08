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
import umap
import ast
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    return df

def generate_embeddings(df, text_column='message'):
    """
    Generate TF-IDF embeddings for text data.
    
    Args:
        df: DataFrame with text data
        text_column: Name of the column containing text data
        
    Returns:
        Numpy array of embeddings
    """
    print(f"Generating TF-IDF embeddings for column '{text_column}'...")
    
    # Check if the text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    # Clean the text data
    texts = df[text_column].fillna('').astype(str).tolist()
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Generate embeddings
    embeddings = vectorizer.fit_transform(texts).toarray()
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Get the feature names for reference
    feature_names = vectorizer.get_feature_names_out()
    print(f"Top features: {', '.join(feature_names[:10])}")
    
    return embeddings, feature_names

def extract_or_generate_embeddings(df, embedding_column='embeddings', text_column='message'):
    """
    Extract embeddings from a DataFrame or generate them if they don't exist.
    
    Args:
        df: DataFrame with data
        embedding_column: Name of the column containing embeddings
        text_column: Name of the column containing text data (used if embeddings don't exist)
        
    Returns:
        DataFrame and numpy array of embeddings
    """
    # Check if the embedding column exists
    if embedding_column in df.columns:
        print(f"Found embedding column '{embedding_column}', extracting embeddings...")
        
        # Extract embeddings
        embeddings = []
        for emb_str in tqdm(df[embedding_column]):
            try:
                # First try to parse as a list
                emb = ast.literal_eval(emb_str)
                embeddings.append(emb)
            except (ValueError, SyntaxError):
                try:
                    # Try to parse as JSON
                    emb = json.loads(emb_str)
                    embeddings.append(emb)
                except:
                    # If all else fails, use a zero vector
                    # Try to infer embedding dimension from previous embeddings
                    if embeddings:
                        dim = len(embeddings[0])
                        embeddings.append([0.0] * dim)
                    else:
                        # Just use a reasonable default dimension
                        embeddings.append([0.0] * 100)
                    print(f"Warning: Could not parse embedding: {emb_str[:50]}...")
        
        # Convert list of embeddings to numpy array
        embeddings_array = np.array(embeddings)
        print(f"Extracted embeddings with shape: {embeddings_array.shape}")
    
    else:
        print(f"No embedding column '{embedding_column}' found, generating embeddings from '{text_column}'...")
        
        # Generate embeddings
        embeddings_array, _ = generate_embeddings(df, text_column)
        
        # Add embeddings to the DataFrame
        df[embedding_column] = [str(list(emb)) for emb in embeddings_array]
    
    return df, embeddings_array

def create_2d_embeddings(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42):
    """
    Create 2D embeddings using UMAP.
    
    Args:
        embeddings: Numpy array of embeddings
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        random_state: Random state for reproducibility
        
    Returns:
        Numpy array of 2D embeddings
    """
    print(f"Creating 2D embeddings with UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    
    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Apply UMAP
    embeddings_2d = reducer.fit_transform(embeddings)
    print(f"Created 2D embeddings with shape: {embeddings_2d.shape}")
    
    return embeddings_2d

def save_results(df, embeddings_2d, output_file=None):
    """
    Save the original data with 2D embeddings to a CSV file.
    
    Args:
        df: Original DataFrame
        embeddings_2d: Numpy array of 2D embeddings
        output_file: Output file path
        
    Returns:
        Path to the saved file
    """
    # Create output file path if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data_with_2d_embeddings_{timestamp}.csv"
    
    print(f"Saving results to {output_file}...")
    
    # Add 2D embeddings to the dataframe
    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]
    
    # Calculate distances from the origin (0,0) as a proxy for anomaly score
    df['distance_from_origin'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Create 2D embeddings for data visualization')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output_file', help='Path to the output CSV file')
    parser.add_argument('--text_column', default='message', help='Name of the column containing text data')
    parser.add_argument('--embedding_column', default='embeddings', help='Name of the column containing embeddings')
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='Minimum distance for UMAP')
    parser.add_argument('--metric', default='cosine', help='Distance metric for UMAP')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input_file)
    
    # Extract or generate embeddings
    df, embeddings = extract_or_generate_embeddings(df, args.embedding_column, args.text_column)
    
    # Create 2D embeddings
    embeddings_2d = create_2d_embeddings(
        embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state
    )
    
    # Save results
    save_results(df, embeddings_2d, args.output_file)

if __name__ == "__main__":
    main() 