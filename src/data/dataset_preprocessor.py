import os
import shutil
import pandas as pd

def validate_file(filepath):
    """
    Validates if a file is suitable for processing.
    Returns (bool, str) tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            return False, "File does not exist"
            
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            return False, "File is empty"
            
        # Try reading as CSV
        df = pd.read_csv(filepath)
        
        # Check if DataFrame is empty
        if df.empty:
            return False, "CSV file has no data"
            
        # Check for required columns ('Sentence' and 'Label')
        required_columns = {'Sentence', 'Label'}
        if not required_columns.issubset(df.columns):
            return False, f"Missing required columns: {required_columns - set(df.columns)}"
            
        return True, "File is valid"
        
    except pd.errors.EmptyDataError:
        return False, "Empty CSV file"
    except pd.errors.ParserError:
        return False, "Invalid CSV format"
    except Exception as e:
        return False, str(e)

def move_invalid_files(filenames, invalid_dir):
    """
    Checks files and moves invalid ones to specified directory.
    Returns list of valid files.
    """
    os.makedirs(invalid_dir, exist_ok=True)
    valid_files = []
    
    for filepath in filenames:
        is_valid, error = validate_file(filepath)
        if is_valid:
            valid_files.append(filepath)
        else:
            print(f"Invalid file {filepath}: {error}")
            dest = os.path.join(invalid_dir, os.path.basename(filepath))
            shutil.move(filepath, dest)
            print(f"Moved to {dest}")
            
    return valid_files

# Usage example:
# invalid_dir = "data/processed/invalid_files"
# valid_files = move_invalid_files(filenames, invalid_dir)
# dataset = SentenceRelationDataset('data/processed/labeled_sentences', valid_files)
# dataset.process()