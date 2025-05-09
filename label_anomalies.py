import pandas as pd
import os
import argparse
import re

def add_anomaly_labels(input_file, output_file=None):
    """
    Add 'is_anomaly' column to a CSV file based on specific conditions:
    - If message contains error indicators OR has_notanf is True, then is_anomaly = True
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
    required_columns = ['has_notanf']
    message_column = None
    
    # Find message column - try common names
    possible_message_cols = ['message', 'log_message', 'text', 'content', 'log_content']
    for col in possible_message_cols:
        if col in df.columns:
            message_column = col
            break
    
    if message_column is None:
        # If no standard message column found, look for any column that might contain text
        for col in df.columns:
            if df[col].dtype == 'object' and 'message' in col.lower():
                message_column = col
                break
                
    missing_columns = []
    if message_column is None:
        print("Warning: Could not find a message column. Will continue with existing columns only.")
    else:
        print(f"Using '{message_column}' as the message column")
        
    if 'has_notanf' not in df.columns:
        missing_columns.append('has_notanf')
        df['has_notanf'] = False
    
    # Function to detect errors in message
    def is_error_message(msg):
        if not isinstance(msg, str):
            return False
            
        # Common error indicators in log messages
        error_patterns = [
            r'\[E\]',                    # [E] tag
            r'[Ee]rror',                 # Error or error
            r'[Ee]xception',             # Exception or exception
            r'[Ff]ail',                  # Fail, failed, failure, etc.
            r'PID=\d+ E',                # PID=12345 E format
            r':[Ee]rror:',               # :Error: or :error:
            r'[Ww]arn',                  # Warning, warning, warn
            r'\[W\]',                    # [W] tag
            r'[Cc]rash',                 # Crash, crashed
            r'[Cc]ritical',              # Critical
            r'[Ff]atal',                 # Fatal
            r'[Tt]imeout',               # Timeout
            r'[Uu]nable to',             # Unable to...
            r'[Ii]nvalid',               # Invalid...
            r'[Nn]ot found',             # Not found
            r'[Dd]enied',                # Denied, Access denied
            r'[Oo]ut of',                # Out of memory, out of space
            r'[Oo]verflow',              # Overflow
            r'[Mm]issing',               # Missing...
            r'[Cc]orrupt'                # Corrupt, corrupted
        ]
        
        # Warning patterns - may be considered less severe
        warning_patterns = [
            r'\[W\]',                    # [W] tag
            r'[Ww]arn',                  # Warning, warning, warn
            r'[Dd]eprecated'             # Deprecated
        ]
        
        # Check if message matches any error pattern
        for pattern in error_patterns:
            if re.search(pattern, msg):
                return True
                
        return False
        
    # Function to detect warnings in message
    def is_warning_message(msg):
        if not isinstance(msg, str):
            return False
            
        # Warning patterns
        warning_patterns = [
            r'\[W\]',                    # [W] tag
            r'[Ww]arn',                  # Warning, warning, warn
            r'[Dd]eprecated'             # Deprecated
        ]
        
        # Check if message matches any warning pattern
        for pattern in warning_patterns:
            if re.search(pattern, msg):
                return True
                
        return False
        
    # Function to detect info messages
    def is_info_message(msg):
        if not isinstance(msg, str):
            return False
            
        # Info patterns
        info_patterns = [
            r'\[I\]',                    # [I] tag
            r'[Ii]nfo',                  # Info
            r'[Ss]tart',                 # Start, started
            r'[Cc]omplete',              # Complete, completed
            r'[Ss]uccess'                # Success, successful
        ]
        
        # Check if message matches any info pattern
        for pattern in info_patterns:
            if re.search(pattern, msg):
                return True
                
        return False
    
    # Add derived columns based on message content
    if message_column:
        # Add message type columns if they don't exist
        df['is_error'] = df[message_column].apply(is_error_message)
        df['is_warning'] = df[message_column].apply(is_warning_message)
        df['is_info'] = df[message_column].apply(is_info_message)
        
        print(f"Analyzed message column '{message_column}' to determine message types:")
        print(f"  - Error messages: {df['is_error'].sum()}")
        print(f"  - Warning messages: {df['is_warning'].sum()}")
        print(f"  - Info messages: {df['is_info'].sum()}")
    
    # Create the is_anomaly column based on conditions
    # An anomaly is: error message OR has_notanf is True
    if message_column:
        df['is_anomaly'] = (df['is_error'] | df['has_notanf'])
    else:
        # If no message column, fall back to existing is_error column if it exists
        if 'is_error' in df.columns:
            df['is_anomaly'] = (df['is_error'] | df['has_notanf'])
        else:
            df['is_anomaly'] = df['has_notanf']  # Only use has_notanf
    
    # Count anomalies
    anomaly_count = df['is_anomaly'].sum()
    total_count = len(df)
    anomaly_percentage = (anomaly_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Identified {anomaly_count} anomalies out of {total_count} records ({anomaly_percentage:.2f}%)")
    print(f"Anomaly distribution by cause:")
    if message_column or 'is_error' in df.columns:
        print(f"  - Error messages: {df['is_error'].sum()} records")
    print(f"  - has_notanf=True: {df['has_notanf'].sum()} records")
    if message_column or ('is_error' in df.columns and 'has_notanf' in df.columns):
        print(f"  - Both conditions: {(df['is_error'] & df['has_notanf']).sum()} records")
    
    # Save the labeled data
    df.to_csv(output_file, index=False)
    print(f"Saved labeled data to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Add anomaly labels to a CSV file')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output_file', help='Output CSV file path (optional)')
    parser.add_argument('-m', '--message_column', help='Column name containing message text (optional)')
    
    args = parser.parse_args()
    
    add_anomaly_labels(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 