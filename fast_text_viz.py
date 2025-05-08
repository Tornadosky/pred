#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import argparse
from tqdm import tqdm
import os
import shutil
from datetime import datetime
import gc  # For garbage collection

def load_data(csv_file, preserve_original=True, nrows=None):
    """Load data from CSV file, optionally creating a copy first"""
    print(f"Loading data from {csv_file}{' (first ' + str(nrows) + ' rows)' if nrows else ''}...")
    
    # If preserve_original is True, create a copy of the file for processing
    working_file = csv_file
    if preserve_original:
        file_name, file_ext = os.path.splitext(csv_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        working_file = f"{file_name}_working_{timestamp}{file_ext}"
        
        print(f"Creating a copy of the original file: {working_file}")
        shutil.copy2(csv_file, working_file)
    
    # Load the data, optionally sampling to reduce size
    df = pd.read_csv(working_file, nrows=nrows)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # If we created a working copy, delete it after loading
    if preserve_original and working_file != csv_file:
        os.remove(working_file)
        print(f"Removed temporary working file: {working_file}")
    
    return df

def get_tfidf_embeddings(texts, max_features=10000):
    """Get TF-IDF embeddings for a list of texts - much faster than BERT"""
    print(f"Generating TF-IDF embeddings with max_features={max_features}...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    embeddings = vectorizer.fit_transform(texts)
    
    # Convert to dense array for UMAP
    embeddings = embeddings.toarray()
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Get the most important features for interpretation
    feature_names = vectorizer.get_feature_names_out()
    print(f"Top features: {', '.join(feature_names[:20])}")
    
    return embeddings, feature_names

def try_import_fasttext():
    """Try to import FastText and gensim"""
    try:
        import gensim
        from gensim.models.fasttext import FastText
        return True
    except ImportError:
        print("FastText not available, will use TF-IDF instead")
        return False

def get_fasttext_embeddings(texts, vector_size=100, epochs=5):
    """Train a FastText model on the texts and get embeddings (intermediate speed/quality)"""
    try:
        from gensim.models.fasttext import FastText
        from gensim.utils import simple_preprocess
    except ImportError:
        print("gensim not installed. Installing...")
        import subprocess
        import sys
        subprocess.call([sys.executable, "-m", "pip", "install", "--user", "gensim"])
        from gensim.models.fasttext import FastText
        from gensim.utils import simple_preprocess
    
    print(f"Processing texts for FastText...")
    # Preprocess the texts
    processed_texts = [simple_preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
    
    # Train FastText model
    print(f"Training FastText model (vector_size={vector_size}, epochs={epochs})...")
    model = FastText(
        processed_texts,
        vector_size=vector_size,
        epochs=epochs,
        min_count=1
    )
    
    # Generate embeddings
    print("Generating FastText embeddings...")
    embeddings = []
    for doc in tqdm(processed_texts, desc="Embedding"):
        # Get vectors for all words in the document and average them
        if doc:  # Skip empty documents
            doc_vector = np.mean([model.wv[word] for word in doc if word in model.wv], axis=0)
            if not np.any(np.isnan(doc_vector)):  # Check for NaN values
                embeddings.append(doc_vector)
            else:
                # Fallback for empty or unknown words
                embeddings.append(np.zeros(vector_size))
        else:
            embeddings.append(np.zeros(vector_size))
    
    embeddings = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings, model.wv.index_to_key[:100]  # Return top words

def reduce_dimensions(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine'):
    """Reduce dimensions of embeddings using UMAP"""
    print(f"Reducing dimensions with UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})...")
    
    # Use low_memory=True for large datasets
    reducer = UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist,
        n_components=n_components, 
        random_state=42,
        metric=metric,
        low_memory=True
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced dimensions to shape: {reduced_embeddings.shape}")
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings, texts=None, n_samples=100, figsize=(12, 10), 
                        output_prefix="text_viz", features=None):
    """Visualize the reduced embeddings"""
    print("Creating visualization...")
    plt.figure(figsize=figsize)
    
    # Plot all points
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                alpha=0.3, s=5, c='blue')
    
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
    
    title = 'Fast Text Embedding Visualization'
    if features:
        feature_str = ", ".join(features[:10])
        title += f"\nTop features: {feature_str}"
    
    plt.title(title)
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
    parser = argparse.ArgumentParser(description='Fast text embedding and visualization')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--data_column', default='data', help='Name of the data column containing text')
    parser.add_argument('--n_neighbors', type=int, default=15, help='n_neighbors parameter for UMAP')
    parser.add_argument('--min_dist', type=float, default=0.1, help='min_dist parameter for UMAP')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of text samples to show in visualization')
    parser.add_argument('--output_prefix', default='fast_text', help='Prefix for output files')
    parser.add_argument('--preserve_original', action='store_true', help='Preserve the original file by working on a copy')
    parser.add_argument('--method', choices=['tfidf', 'fasttext'], default='tfidf', help='Embedding method (tfidf is faster, fasttext is better quality)')
    parser.add_argument('--max_features', type=int, default=5000, help='Maximum features for TF-IDF')
    parser.add_argument('--vector_size', type=int, default=100, help='Vector size for FastText')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs for FastText')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size to process (for large files)')
    
    args = parser.parse_args()
    
    # Load data, optionally with sampling for large files
    df = load_data(args.csv_file, preserve_original=args.preserve_original, nrows=args.sample_size)
    
    if args.data_column not in df.columns:
        print(f"Error: Column '{args.data_column}' not found in CSV. Available columns: {', '.join(df.columns)}")
        return
    
    # Get texts from data column
    texts = df[args.data_column].fillna('').astype(str).tolist()
    
    # Generate embeddings based on chosen method
    features = None
    if args.method == 'tfidf':
        embeddings, features = get_tfidf_embeddings(texts, max_features=args.max_features)
    elif args.method == 'fasttext':
        if try_import_fasttext():
            embeddings, features = get_fasttext_embeddings(texts, vector_size=args.vector_size, epochs=args.epochs)
        else:
            print("Falling back to TF-IDF method.")
            embeddings, features = get_tfidf_embeddings(texts, max_features=args.max_features)
    
    # Free memory
    del df
    gc.collect()
    
    # Reduce dimensions with UMAP
    reduced_embeddings = reduce_dimensions(embeddings, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    
    # Free more memory
    del embeddings
    gc.collect()
    
    # Visualize embeddings
    visualize_embeddings(reduced_embeddings, texts, n_samples=args.n_samples, 
                        output_prefix=args.output_prefix, features=features)

if __name__ == "__main__":
    import sys
    main() 