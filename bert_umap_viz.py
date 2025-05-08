#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import json
import umap
import ast
import re
from datetime import datetime
from tqdm import tqdm
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(file_path, n_samples=None):
    """
    Load data from a CSV file containing BERT embeddings.
    
    Args:
        file_path: Path to the CSV file
        n_samples: Number of samples to load (None for all)
        
    Returns:
        DataFrame with the data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # Sample data if requested
    if n_samples is not None and n_samples < len(df):
        print(f"Sampling {n_samples} rows from the dataset")
        df = df.sample(n_samples, random_state=42)
    
    return df

def extract_bert_embeddings(df, embedding_column='embeddings', data_column=None, batch_size=1000):
    """
    Extract BERT embeddings from the DataFrame.
    
    Args:
        df: DataFrame with data
        embedding_column: Name of the column containing BERT embeddings
        data_column: If specified, extract embeddings from a nested JSON in this column
        batch_size: Size of batches for processing
        
    Returns:
        Numpy array of BERT embeddings
    """
    print(f"Extracting embeddings...")
    
    # Check embedding source
    if data_column is not None and data_column in df.columns:
        print(f"Extracting embeddings from nested data in '{data_column}' column")
        
        # Extract embeddings from the nested JSON in data column
        embeddings = []
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                try:
                    # Parse the data column as JSON
                    data_obj = json.loads(row[data_column])
                    
                    # Extract embedding from the data object
                    if embedding_column in data_obj:
                        emb = data_obj[embedding_column]
                        embeddings.append(emb)
                    else:
                        # If embedding not found, use zeros
                        if embeddings:
                            dim = len(embeddings[0])
                            embeddings.append([0.0] * dim)
                        else:
                            # Default embedding dimension
                            embeddings.append([0.0] * 768)
                        print(f"Warning: No '{embedding_column}' in data object")
                except Exception as e:
                    # If parsing fails, use zeros
                    if embeddings:
                        dim = len(embeddings[0])
                        embeddings.append([0.0] * dim)
                    else:
                        # Default embedding dimension
                        embeddings.append([0.0] * 768)
                    print(f"Warning: Error parsing data column: {str(e)}")
    
    elif embedding_column in df.columns:
        print(f"Extracting embeddings from '{embedding_column}' column")
        
        # Extract embeddings directly from the embedding column
        embeddings = []
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            for emb_str in batch[embedding_column]:
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
                        # If all else fails, use zeros
                        if embeddings:
                            dim = len(embeddings[0])
                            embeddings.append([0.0] * dim)
                        else:
                            # Default embedding dimension
                            embeddings.append([0.0] * 768)
                        print(f"Warning: Could not parse embedding: {emb_str[:50]}...")
    
    # If no embedding column or data column found with embeddings, try to generate them from the message
    elif 'message' in df.columns:
        print("No embeddings found. Generating embeddings from 'message' column...")
        # This would use a basic TF-IDF to generate embeddings
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=100)
        messages = df['message'].astype(str).fillna('')
        embeddings = vectorizer.fit_transform(messages).toarray()
        print(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
        return embeddings
    
    else:
        # Find a text column to use for embeddings
        text_columns = [col for col in df.columns if col.lower() in ['text', 'log', 'message', 'body']]
        
        if text_columns:
            text_col = text_columns[0]
            print(f"No embeddings found. Generating embeddings from '{text_col}' column...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=100)
            texts = df[text_col].astype(str).fillna('')
            embeddings = vectorizer.fit_transform(texts).toarray()
            print(f"Generated TF-IDF embeddings with shape: {embeddings.shape}")
            return embeddings
        else:
            raise ValueError(f"Neither embedding column '{embedding_column}' nor data column '{data_column}' found in DataFrame, and no text column found to generate embeddings. Available columns: {', '.join(df.columns)}")
    
    # Convert list of embeddings to numpy array
    embeddings_array = np.array(embeddings)
    print(f"Extracted embeddings with shape: {embeddings_array.shape}")
    
    return embeddings_array

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

def save_results(df, embeddings_2d, output_prefix=None, preserve_original=False):
    """
    Save the original data with 2D embeddings to a CSV file.
    
    Args:
        df: Original DataFrame
        embeddings_2d: Numpy array of 2D embeddings
        output_prefix: Prefix for output files
        preserve_original: Whether to keep all original columns in the output
        
    Returns:
        Path to the saved file
    """
    # Create output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "bert_umap_output"
    os.makedirs(output_dir, exist_ok=True)
    
    if output_prefix:
        output_file = os.path.join(output_dir, f"{output_prefix}_{timestamp}.csv")
    else:
        output_file = os.path.join(output_dir, f"embeddings_2d_{timestamp}.csv")
    
    print(f"Saving results to {output_file}...")
    
    if preserve_original:
        # Keep all original columns and add the 2D embeddings
        output_df = df.copy()
        output_df['x'] = embeddings_2d[:, 0]
        output_df['y'] = embeddings_2d[:, 1]
    else:
        # Create a new DataFrame with just the essential columns
        output_df = pd.DataFrame()
        
        # Try to find useful columns to preserve
        important_columns = ['_time', 'message', 'log', 'text', 'data', 'has_notanf']
        for col in important_columns:
            if col in df.columns:
                output_df[col] = df[col]
        
        # Add the 2D embeddings
        output_df['x'] = embeddings_2d[:, 0]
        output_df['y'] = embeddings_2d[:, 1]
    
    # Calculate distances from the origin (0,0) as a proxy for anomaly score
    output_df['distance_from_origin'] = np.sqrt(output_df['x']**2 + output_df['y']**2)
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Convert embeddings to 2D using UMAP for visualization')
    parser.add_argument('csv_file', help='Path to the input CSV file containing embeddings')
    parser.add_argument('--data_column', default=None, help='Name of the column containing nested JSON data with embeddings')
    parser.add_argument('--embedding_column', default='embeddings', help='Name of the column containing embeddings')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--n_neighbors', type=int, default=15, help='Number of neighbors for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='Minimum distance for UMAP')
    parser.add_argument('--n_samples', type=int, help='Number of samples to process (None for all)')
    parser.add_argument('--output_prefix', help='Prefix for output files')
    parser.add_argument('--preserve_original', action='store_true', help='Preserve all original columns in output')
    parser.add_argument('--use_sentence_transformer', action='store_true', help='Use SentenceTransformer for embedding')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data(args.csv_file, args.n_samples)
        
        # For cleaned_logs_bigger.csv, use the 'data' column by default
        if '_time' in df.columns and 'data' in df.columns and args.data_column is None:
            print("Found '_time' and 'data' columns, assuming this is cleaned_logs_bigger.csv")
            args.data_column = 'data'
        
        # Extract embeddings
        embeddings = extract_bert_embeddings(
            df, 
            embedding_column=args.embedding_column,
            data_column=args.data_column,
            batch_size=args.batch_size
        )
        
        # Create 2D embeddings
        embeddings_2d = create_2d_embeddings(
            embeddings,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric='cosine',
            random_state=42
        )
        
        # Save results
        save_results(
            df, 
            embeddings_2d, 
            output_prefix=args.output_prefix,
            preserve_original=args.preserve_original
        )
        
        print("Process completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main() 