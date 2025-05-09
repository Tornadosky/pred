import pandas as pd
import os
import argparse

def add_anomaly_labels(input_file, output_file=None):
    """
    Add 'is_anomaly' column to a CSV file based on specific conditions:
    - If is_error is True OR has_notanf is True, then is_anomaly = True
    - Otherwise, is_anomaly = False
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to save the output CSV file. If None, 
                                     a default name will be generated
    """
    print(f"Processing file: {input_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_labeled{ext}"
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check if required columns exist
    missing_columns = []
    if 'is_error' not in df.columns:
        missing_columns.append('is_error')
    if 'has_notanf' not in df.columns:
        missing_columns.append('has_notanf')
    
    if missing_columns:
        print(f"Warning: Missing required columns: {', '.join(missing_columns)}")
        print("Adding missing columns with default value False")
        for col in missing_columns:
            df[col] = False
    
    # Create the is_anomaly column based on conditions
    df['is_anomaly'] = (df['is_error'] | df['has_notanf'])
    
    # Count anomalies
    anomaly_count = df['is_anomaly'].sum()
    total_count = len(df)
    anomaly_percentage = (anomaly_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Identified {anomaly_count} anomalies out of {total_count} records ({anomaly_percentage:.2f}%)")
    print(f"Anomaly distribution by cause:")
    print(f"  - is_error=True: {df['is_error'].sum()} records")
    print(f"  - has_notanf=True: {df['has_notanf'].sum()} records")
    print(f"  - Both conditions: {(df['is_error'] & df['has_notanf']).sum()} records")
    
    # Save the labeled data
    df.to_csv(output_file, index=False)
    print(f"Saved labeled data to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Add anomaly labels to a CSV file')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output_file', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    add_anomaly_labels(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 