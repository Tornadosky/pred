#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime
from tqdm import tqdm
import warnings

# For dimensionality reduction and clustering
from umap import UMAP
import hdbscan
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(file_path, n_samples=None, bert_columns_prefix='bert_dim_'):
    """
    Load data from a CSV file containing BERT embeddings.
    
    Args:
        file_path: Path to the CSV file
        n_samples: Number of samples to load (None for all)
        bert_columns_prefix: Prefix for BERT embedding dimension columns
        
    Returns:
        DataFrame with the data, DataFrame with just BERT embeddings
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows with columns: {', '.join(df.columns)}")
    
    # Sample data if requested
    if n_samples is not None and n_samples < len(df):
        print(f"Sampling {n_samples} rows from the dataset")
        df = df.sample(n_samples, random_state=42)
    
    # Extract BERT embedding columns
    bert_cols = [col for col in df.columns if col.startswith(bert_columns_prefix)]
    if not bert_cols:
        if 'bert_embeddings' in df.columns:
            print("Found 'bert_embeddings' column, parsing JSON strings...")
            # Parse JSON strings into separate columns
            bert_embeddings = []
            for emb_str in tqdm(df['bert_embeddings'], desc="Parsing embeddings"):
                try:
                    emb = json.loads(emb_str)
                    bert_embeddings.append(emb)
                except:
                    print(f"Error parsing embedding: {emb_str[:50]}...")
                    # Use zeros as fallback
                    if bert_embeddings:
                        dim = len(bert_embeddings[0])
                        bert_embeddings.append([0.0] * dim)
                    else:
                        # Default embedding dimension for BERT
                        bert_embeddings.append([0.0] * 768)
            
            # Convert to numpy array
            bert_array = np.array(bert_embeddings)
            
            # Create a DataFrame with the parsed embeddings
            bert_df = pd.DataFrame(
                bert_array, 
                columns=[f'bert_dim_{i}' for i in range(bert_array.shape[1])]
            )
            
            # Update the original DataFrame with the parsed embeddings
            for col in bert_df.columns:
                df[col] = bert_df[col].values
                
            bert_cols = bert_df.columns.tolist()
        else:
            raise ValueError(f"No BERT embedding columns found with prefix '{bert_columns_prefix}' and no 'bert_embeddings' column found")
    
    print(f"Found {len(bert_cols)} BERT embedding dimensions")
    bert_df = df[bert_cols]
    
    return df, bert_df

def extract_additional_features(df):
    """
    Extract additional features from the dataframe to combine with BERT embeddings.
    
    Args:
        df: Original DataFrame
        
    Returns:
        DataFrame with additional features
    """
    print("Extracting additional features...")
    
    # Create an empty DataFrame for the additional features
    features = pd.DataFrame(index=df.index)
    
    # Identify available feature columns
    # First, check for the standard feature columns from the findata.csv format
    standard_feature_cols = [
        'number_of_day', 'sst_number', 'has_lsasst', 'has_notanf', 
        'has_numeric_data', 'has_OKAY_word', 'message_length',
        'is_info', 'is_warning', 'is_error',
        'operation_type_spool', 'operation_type_acknowledge', 
        'operation_type_receive', 'operation_type_other', 'operation_type_send'
    ]
    
    # Filter to only include columns that exist in the dataframe
    available_feature_cols = [col for col in standard_feature_cols if col in df.columns]
    
    if available_feature_cols:
        # Use the identified feature columns
        features = df[available_feature_cols].copy()
        print(f"Using {len(available_feature_cols)} standard feature columns: {', '.join(available_feature_cols)}")
    else:
        # No standard features found, try to extract features from text if possible
        if 'message' in df.columns:
            print("Standard features not found. Extracting features from 'message' column...")
            features['message_length'] = df['message'].astype(str).str.len()
            features['has_error_word'] = df['message'].astype(str).str.contains('error|fail|exception', case=False).astype(int)
            features['has_warning_word'] = df['message'].astype(str).str.contains('warn|caution', case=False).astype(int)
            features['has_info_word'] = df['message'].astype(str).str.contains('info|log|notice', case=False).astype(int)
            features['has_numbers'] = df['message'].astype(str).str.contains(r'\d+').astype(int)
        else:
            print("No standard features or message column found. Using only BERT embeddings.")
    
    # Fill missing values with zeros
    features = features.fillna(0)
    
    print(f"Extracted {features.shape[1]} additional features")
    return features

def combine_features(bert_df, additional_features=None, use_additional_features=True, scale=True):
    """
    Combine BERT embeddings with additional features.
    
    Args:
        bert_df: DataFrame with BERT embeddings
        additional_features: DataFrame with additional features
        use_additional_features: Whether to include additional features
        scale: Whether to scale the features
        
    Returns:
        Combined feature matrix
    """
    # Start with BERT embeddings
    if scale:
        print("Scaling BERT embeddings...")
        bert_scaler = StandardScaler()
        bert_features = bert_scaler.fit_transform(bert_df)
    else:
        bert_features = bert_df.values
    
    # Combine with additional features if available and requested
    if additional_features is not None and use_additional_features and not additional_features.empty:
        print(f"Combining BERT embeddings with {additional_features.shape[1]} additional features...")
        if scale:
            feat_scaler = StandardScaler()
            additional_features_scaled = feat_scaler.fit_transform(additional_features)
        else:
            additional_features_scaled = additional_features.values
        
        # Combine the scaled features
        combined_features = np.hstack((bert_features, additional_features_scaled))
        print(f"Combined feature matrix shape: {combined_features.shape}")
    else:
        combined_features = bert_features
        print(f"Using only BERT embeddings with shape: {combined_features.shape}")
    
    return combined_features

def perform_umap_reduction(features, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Perform dimensionality reduction using UMAP.
    
    Args:
        features: Feature matrix
        n_components: Number of dimensions to reduce to
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        
    Returns:
        Reduced feature matrix
    """
    print(f"Performing UMAP reduction to {n_components} dimensions...")
    umap_reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    
    umap_embedding = umap_reducer.fit_transform(features)
    print(f"UMAP embedding shape: {umap_embedding.shape}")
    
    return umap_embedding

def perform_hdbscan_clustering(embedding, min_cluster_size=15, min_samples=5, metric='euclidean'):
    """
    Perform clustering using HDBSCAN.
    
    Args:
        embedding: Reduced feature matrix (typically from UMAP)
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples for a core point
        metric: Distance metric for HDBSCAN
        
    Returns:
        Cluster labels, outlier scores
    """
    print(f"Performing HDBSCAN clustering (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        gen_min_span_tree=True,  # Required for outlier scores
        prediction_data=True     # Required for outlier scores
    )
    
    cluster_labels = clusterer.fit_predict(embedding)
    
    # Calculate outlier scores (higher = more likely to be an outlier)
    outlier_scores = clusterer.outlier_scores_
    
    # Count the number of clusters and noise points
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters")
    print(f"Identified {n_noise} noise points ({n_noise/len(cluster_labels):.1%} of data)")
    
    return cluster_labels, outlier_scores

def visualize_clusters(embedding, cluster_labels, outlier_scores, original_df=None, cluster_info=None, color_by='cluster'):
    """
    Create interactive visualizations of the clusters using Plotly.
    
    Args:
        embedding: 2D embedding for visualization
        cluster_labels: Cluster labels from HDBSCAN
        outlier_scores: Outlier scores from HDBSCAN
        original_df: Original DataFrame for hover information
        cluster_info: Additional information about clusters
        color_by: Column to use for coloring points ('cluster' or a column name in original_df)
        
    Returns:
        Plotly figure
    """
    print(f"Creating visualization colored by '{color_by}'...")
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': cluster_labels,
        'outlier_score': outlier_scores
    })
    
    # Add data from original DataFrame for hover information and coloring
    if original_df is not None:
        # Add message text if available
        if 'message' in original_df.columns:
            plot_df['message'] = original_df['message'].values
        
        # Copy relevant columns for hover data
        hover_columns = ['is_error', 'is_warning', 'is_info', 'has_notanf', 
                        'has_lsasst', 'has_numeric_data', 'has_OKAY_word', 
                        'operation_type_spool', 'operation_type_acknowledge',
                        'operation_type_receive', 'operation_type_send']
        
        for col in hover_columns:
            if col in original_df.columns:
                plot_df[col] = original_df[col].values
    
    # Determine hover data columns
    hover_data = ['outlier_score', 'cluster']
    if 'message' in plot_df.columns:
        hover_data.append('message')
    
    # Add any additional relevant columns to hover data
    for col in plot_df.columns:
        if col not in ['x', 'y', 'cluster', 'outlier_score', 'message'] and col in hover_columns:
            hover_data.append(col)
    
    # Initialize color settings with defaults
    color_column = 'cluster'
    color_continuous_scale = px.colors.qualitative.G10
    color_discrete_map = None
    title = 'HDBSCAN Clustering of BERT Embeddings'
    
    # Determine the color column and settings
    if color_by == 'cluster':
        # Default cluster coloring - already set above
        pass
    elif color_by in plot_df.columns:
        # Color by a specific column
        color_column = color_by
        # Use discrete coloring for binary columns
        if plot_df[color_by].nunique() <= 2 and set(plot_df[color_by].unique()).issubset({0, 1}):
            color_discrete_map = {0: 'lightblue', 1: 'red'}
            color_continuous_scale = None  # Not used for discrete coloring
            title = f'BERT Embeddings Colored by {color_by}'
        else:
            color_continuous_scale = 'Viridis'
            color_discrete_map = None
            title = f'BERT Embeddings Colored by {color_by}'
    else:
        # Fallback to cluster coloring if requested column doesn't exist
        print(f"Warning: Column '{color_by}' not found, coloring by cluster instead")
    
    # Create the scatter plot with appropriate color settings
    if color_discrete_map is not None:
        # Discrete coloring (for binary features)
        fig = px.scatter(
            plot_df, 
            x='x', 
            y='y',
            color=color_column,
            hover_data=hover_data,
            title=title,
            color_discrete_map=color_discrete_map,
            labels={'cluster': 'Cluster', color_column: color_column.replace('_', ' ').title()}
        )
    else:
        # Continuous coloring
        fig = px.scatter(
            plot_df, 
            x='x', 
            y='y',
            color=color_column,
            hover_data=hover_data,
            title=title,
            color_continuous_scale=color_continuous_scale,
            labels={'cluster': 'Cluster', color_column: color_column.replace('_', ' ').title()}
        )
    
    # Update the figure layout
    fig.update_layout(
        template='plotly_white',
        width=1000,
        height=800,
        legend_title_text=color_column.replace('_', ' ').title(),
    )
    
    # Update marker size and opacity
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=1, color='DarkSlateGrey')
        )
    )
    
    return fig

def save_results(df, embedding, cluster_labels, outlier_scores, output_dir='viz_output'):
    """
    Save clustering results to CSV and visualization to HTML.
    
    Args:
        df: Original DataFrame
        embedding: 2D embedding
        cluster_labels: Cluster labels
        outlier_scores: Outlier scores
        output_dir: Directory to save results
        
    Returns:
        Path to the saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['x'] = embedding[:, 0]
    results_df['y'] = embedding[:, 1]
    results_df['cluster'] = cluster_labels
    results_df['outlier_score'] = outlier_scores
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, f"hdbscan_clusters_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create and save primary visualization (colored by cluster)
    html_paths = []
    try:
        fig_cluster = visualize_clusters(embedding, cluster_labels, outlier_scores, original_df=df, color_by='cluster')
        html_cluster_path = os.path.join(output_dir, f"hdbscan_visualization_{timestamp}.html")
        fig_cluster.write_html(html_cluster_path)
        html_paths.append(html_cluster_path)
        print(f"Cluster visualization saved to {html_cluster_path}")
    except Exception as e:
        print(f"Error creating cluster visualization: {str(e)}")
    
    # Create additional visualizations colored by different features
    feature_columns = ['is_error', 'is_warning', 'is_info', 'has_notanf', 'has_lsasst', 'has_numeric_data']
    
    for feature in feature_columns:
        if feature in df.columns:
            try:
                fig_feature = visualize_clusters(embedding, cluster_labels, outlier_scores, original_df=df, color_by=feature)
                html_feature_path = os.path.join(output_dir, f"visualization_by_{feature}_{timestamp}.html")
                fig_feature.write_html(html_feature_path)
                html_paths.append(html_feature_path)
                print(f"Visualization colored by {feature} saved to {html_feature_path}")
            except Exception as e:
                print(f"Error creating visualization for {feature}: {str(e)}")
    
    return csv_path, html_paths

def main():
    parser = argparse.ArgumentParser(description='Perform HDBSCAN clustering on data with BERT embeddings')
    parser.add_argument('csv_file', help='Path to the CSV file with BERT embeddings')
    parser.add_argument('--bert_columns_prefix', default='bert_dim_', 
                        help='Prefix for BERT embedding column names (default: bert_dim_)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Number of samples to use (default: all)')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP (default: 15)')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                        help='Minimum distance for UMAP (default: 0.1)')
    parser.add_argument('--min_cluster_size', type=int, default=15,
                        help='Minimum cluster size for HDBSCAN (default: 15)')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples for HDBSCAN (default: 5)')
    parser.add_argument('--output_dir', default='viz_output',
                        help='Directory to save results (default: viz_output)')
    parser.add_argument('--no_additional_features', action='store_true',
                        help='Use only BERT embeddings without additional features')
    parser.add_argument('--additional_colors', nargs='+', default=[],
                        help='Additional columns to create color-coded visualizations for')
    
    args = parser.parse_args()
    
    # Load the data
    df, bert_df = load_data(args.csv_file, n_samples=args.n_samples, bert_columns_prefix=args.bert_columns_prefix)
    
    # Extract additional features
    additional_features = extract_additional_features(df)
    
    # Combine features
    combined_features = combine_features(bert_df, additional_features, not args.no_additional_features)
    
    # Perform dimensionality reduction
    umap_embedding = perform_umap_reduction(
        combined_features, 
        n_neighbors=args.umap_neighbors, 
        min_dist=args.umap_min_dist
    )
    
    # Perform clustering
    cluster_labels, outlier_scores = perform_hdbscan_clustering(
        umap_embedding, 
        min_cluster_size=args.min_cluster_size, 
        min_samples=args.min_samples
    )
    
    # Save results and create visualizations
    save_results(df, umap_embedding, cluster_labels, outlier_scores, output_dir=args.output_dir)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 