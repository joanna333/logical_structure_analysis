import re
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import glob
import os


class TextPreprocessor:
    valid_labels = [ "Causal", "Conditional", "Sequential", "Comparison", "Explanation", 
                    "Definition", "Contrast", "Addition", "Emphasis", "Elaboration", 
                    "Illustration", "Concession", "Generalization", "Inference", 
                    "Summary", "Problem Solution", "Contrastive Emphasis", "Purpose", 
                    "Clarification", "Enumeration", "Cause and Effect", "Temporal Sequence" ]
    def split_raw_text(self, text: str) -> None:
        """Split text into sentences and relationships and save to CSV.
        
        Args:
            text: Input text with relationship markers
            output_file: Path to save CSV output
        """
        # Split text by new line and remove empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Extract sentences and relationships
        data = []
        for line in lines:
            # Find relationship marker if exists
            relationship_match = re.search(r'-\[(\w+)\]-?', line)
            if relationship_match:
                # Get relationship type and clean sentence
                relationship = relationship_match.group(1)
                sentence = re.sub(r'-\[\w+\]-?', '', line).strip()
                data.append({
                    'Sentence': sentence,
                    'Label': relationship
                })
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        return df

    def clean_text(self, text: str) -> str:
        """Remove markdown formatting, quotes, and relationship markers from text"""
        # Remove markdown and quotes
        cleaned_text = text.replace('**', '').replace('"', ' ')
        
        # Remove relationship markers with updated regex to match also -[Cause and Effect]-
        cleaned_text = re.sub(r'-\[[\w\s]+\]-', '', cleaned_text)  # Matches -[Definition]-, -[Causal]-, -[Cause and Effect]- etc.
        cleaned_text = re.sub(r'-\[[\w\s]+\]', '', cleaned_text)   # Matches -[Definition], -[Causal], -[Cause and Effect] etc.
        
        return cleaned_text

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences and clean them"""
        cleaned_text = self.clean_text(text)
        sentences = cleaned_text.replace("\n", ". ").split(".")
        sentences = [sentence.strip() for sentence in sentences 
                    if sentence.strip() and not sentence.strip().isdigit()]
        return sentences
    
    # # Combine all files into one dataframe and use file names as index
    # def combine_files(self, files: List[str]) -> pd.DataFrame:
    #     """Combine multiple CSV files into one DataFrame, filtering by specific labels"""
    #     valid_labels = [
    #         "Causal", "Conditional", "Sequential", "Comparison", "Explanation", "Definition", 
    #         "Contrast", "Addition", "Emphasis", "Elaboration", "Illustration", "Concession", 
    #         "Generalization", "Inference", "Summary", "Problem Solution", "Contrastive Emphasis", 
    #         "Purpose", "Clarification", "Enumeration", "Cause and Effect", "Temporal Sequence"
    #     ]
        
    #     dfs = []
    #     for file in files:
    #         df = pd.read_csv(file, encoding='latin-1')
    #         # Clean labels
    #         df['Label'] = df['Label'].apply(self.clean_label)
    #         # Filter by valid labels
    #         df = df[df['Label'].isin(valid_labels)]
    #         dfs.append(df)
        
    #     combined_df = pd.concat(dfs, ignore_index=True)
    #     combined_df = combined_df[combined_df['Label'].isin(valid_labels)]
        
    #     return combined_df
    def combine_files(self, files):
        """Combine multiple CSV files with error handling"""
        combined_df = pd.DataFrame()
        
        for file in files:
            try:
                print(f"Processing file: {file}")
                
                # Skip .DS_Store files
                if '.DS_Store' in file:
                    print(f"Skipping system file: {file}")
                    continue
                    
                # Read CSV file
                df = pd.read_csv(file, encoding='latin-1')
                print(f"Columns found: {df.columns.tolist()}")
                
                # Skip files without required columns
                if 'Label' not in df.columns:
                    print(f"Warning: No 'Label' column found in {file}")
                    continue
                    
                # Verify DataFrame is not empty
                if df.empty:
                    print(f"Warning: Empty DataFrame in {file}")
                    continue
                    
                # Concatenate with existing data
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
                
        # Verify the combined DataFrame has required columns
        if not combined_df.empty and 'Label' in combined_df.columns:
            # Clean labels in the combined DataFrame
            combined_df['Label'] = combined_df['Label'].apply(self.clean_label)
        else:
            print("Warning: No valid data found to combine")
            return pd.DataFrame(columns=['Sentence', 'Label'])
            
        return combined_df

    # Export dataframe to CSV
    def export_csv(self, df: pd.DataFrame, output_file: str) -> None:
        """Export DataFrame to CSV"""
        df.to_csv(output_file, index=False)

     # Find all files names from a folder
    def find_files(self, folder: str) -> List[str]:
        """Find all files in a folder"""
        return [str(file) for file in Path(folder).rglob("*") if file.is_file()]
    
    # convert all text files in  directory to csv
    def convert_to_csv(self, folder: str) -> None:
        """Convert all text files in a folder to CSV"""
        files = self.find_files(folder)
        for file in files:
            with open(file, "r") as f:
                text = f.read()
            df = self.split_raw_text(text)

            output_file = f"{file}.csv"
            self.export_csv(df, output_file)
    # include .csv at the end of the file name for all files in a directory
    def add_csv_extension(self, folder: str) -> None:
        """Add .csv extension to all files in a folder"""
        files = self.find_files(folder)
        for file in files:
            new_file = f"{file}.csv"
            Path(file).rename(new_file)
    # count all lines in all files in a directory and substratct the number of files
    def count_lines(self, folder: str) -> int:
        """Count total lines in all files in a folder"""
        files = self.find_files(folder)
        total_lines = 0
        for file in files:
            with open(file, 'r', encoding='latin-1') as f:
                total_lines += len(f.readlines())
        return total_lines - len(files)
    
    # Give the count of a specific label in a column in a dataframe
    def count_label(self, df: pd.DataFrame, column: str, label: str) -> int:
        """Count occurrences of a label in a column of a DataFrame"""
        return df[column].str.count(label).sum()
    
    # Give the count of a specific label in all files in a directory
    def count_label_in_files(self, folder: str, label: str) -> int:
        """Count occurrences of a label in all files in a folder"""
        files = self.find_files(folder)
        total_count = 0
        for file in files:
            df = pd.read_csv(file, encoding='latin-1')
            total_count += self.count_label(df, 'Label', label)
        return total_count
    
    # Give the count of all lables in all files in a directory
    def count_all_labels_in_files(self, directory: str) -> Dict[str, int]:
        """Count occurrences of each label across all CSV files.
        
        Args:
            directory: Directory containing CSV files
            
        Returns:
            Dictionary mapping labels to counts
        """
        label_counts = {}
        for file in glob.glob(os.path.join(directory, "*.csv")):
            try:
                df = pd.read_csv(file, encoding='latin-1')
                if not self.validate_dataframe(df):
                    continue
                    
                # Clean labels before counting
                df['Label'] = df['Label'].apply(self.clean_label)
                
                for label in df['Label'].unique():
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += len(df[df['Label'] == label])
                    
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                
        return label_counts

    def validate_dataframe(self, df):
        """Validate DataFrame structure and content"""
        if len(df.columns) != 2:
            print(f"Warning: Expected 2 columns, found {len(df.columns)}")
            return False
            
        expected_columns = {'Sentence', 'Label'}
        if not set(df.columns).issuperset(expected_columns):
            print(f"Warning: Missing required columns. Found: {df.columns}")
            return False
            
        return True
    
    # Export the count of different labels in all files in a directory to a csv file
    def export_label_counts(self, directory, output_file):
        label_counts = self.count_all_labels_in_files(directory)
        df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
        df.to_csv(output_file, index=False)

    def clean_label(self, label: str) -> str:
        """Clean label by removing extra spaces and quotation marks.
        
        Args:
            label: Label string to clean
            
        Returns:
            Cleaned label string
        """
        # Remove quotes and spaces
        cleaned = label.strip().strip('"').strip()
        # Replace multiple spaces with single space
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def process_csv_labels(self, file_path: str) -> None:
        """Process CSV file to clean label formatting.
        
        Args:
            file_path: Path to CSV file to process
        """
        try:
            # Read CSV
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Validate columns
            if not self.validate_dataframe(df):
                print(f"Skipping invalid file: {file_path}")
                return
                
            # Clean labels
            df['Label'] = df['Label'].apply(self.clean_label)
            
            # Write back to file
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    def process_directory_labels(self, directory: str) -> None:
        """Process all CSV files in directory to clean labels.
        
        Args:
            directory: Directory containing CSV files
        """
        for file in glob.glob(os.path.join(directory, "*.csv")):
            print(f"Processing {file}")
            self.process_csv_labels(file)

    # Find the file name and line number of a specific label in a list, and return the file name and line number
    # Check all files in a directory
    def find_label(self, directory: str, label: str) -> List[Tuple[str, int]]:
        """Find occurrences of a label in all files in a directory.
        
        Args:
            directory: Directory containing CSV files
            label: Label to search for
            
        Returns:
            List of tuples containing file name and line number
        """
        files = self.find_files(directory)
        matches = []
        for file in files:
            with open(file, 'r', encoding='latin-1') as f:
                for i, line in enumerate(f.readlines(), 1):
                    if label in line:
                        matches.append((file, i))
        return matches

    def count_labels_in_combined_df(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count occurrences of each label in combined DataFrame.
        
        Args:
            df: Combined DataFrame with 'Label' column
            
        Returns:
            Dictionary of label counts
        """
        if df.empty or 'Label' not in df.columns:
            return {}
            
        # Count labels and convert to dictionary
        label_counts = df['Label'].value_counts().to_dict()
        return label_counts

    def combine_and_count_files(self, directory: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Combine files and count labels from a directory.
        
        Args:
            directory: Directory containing CSV files
            
        Returns:
            Tuple of (combined DataFrame, label counts dict)
        """
        # Verify directory exists
        if not os.path.exists(directory):
            print(f"Error: Directory not found: {directory}")
            return pd.DataFrame(), {}

        # Get list of CSV files
        files = glob.glob(os.path.join(directory, "*.csv"))
        if not files:
            print(f"No CSV files found in {directory}")
            return pd.DataFrame(), {}

        # Combine files
        combined_df = pd.DataFrame()
        for file in files:
            try:
                df = pd.read_csv(file, encoding='latin-1')
                if 'Label' not in df.columns:
                    print(f"Warning: No Label column in {file}")
                    continue
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

        # Count labels if we have data
        if not combined_df.empty and 'Label' in combined_df.columns:
            label_counts = combined_df['Label'].value_counts().to_dict()
        else:
            label_counts = {}

        # Export counts
        output_file = os.path.join(directory, f"label_counts_{os.path.basename(directory)}.csv")
        if label_counts:
            counts_df = pd.DataFrame(list(label_counts.items()), columns=['Label', 'Count'])
            counts_df.to_csv(output_file, index=False)

        return combined_df, label_counts
    def count_unique_labels_in_file(self, file_path: str, output_file: str) -> Dict[str, int]:
        """Count unique labels in a single CSV file and export to CSV.
        
        Args:
            file_path: Path to CSV file
            output_file: Path to save the output CSV file
            
        Returns:
            Dictionary mapping labels to their counts
        """
        # Validate file exists and is CSV
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return {}
            
        if not file_path.endswith('.csv'):
            print(f"Error: Not a CSV file: {file_path}")
            return {}
            
        try:
            # Read CSV
            df = pd.read_csv(file_path, encoding='latin-1')
            
            # Verify Label column exists
            if 'Label' not in df.columns:
                print(f"Error: No Label column in {file_path}")
                return {}
                
            # Count unique labels
            label_counts = df['Label'].value_counts().to_dict()
            
            # Clean labels if needed
            clean_counts = {self.clean_label(k): v for k, v in label_counts.items()}
            
            # Export to CSV
            counts_df = pd.DataFrame(list(clean_counts.items()), columns=['Label', 'Count'])
            counts_df.to_csv(output_file, index=False)
            
            return clean_counts
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {}

# Update main execution:
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    # # Process each directory
    # base_dir = "data/processed"
    # dirs = ['labeled_sentences', 'new_labeled_sentences', 'sentence_types']
    
    # for dir_name in dirs:
    #     print(f"\nProcessing {dir_name}...")
    #     full_path = os.path.join(base_dir, dir_name)
    #     combined_df, counts = preprocessor.combine_and_count_files(full_path)
    #     print(f"Found {len(combined_df)} rows in {dir_name}")
    #     print(f"Label counts: {counts}")
counts = preprocessor.count_unique_labels_in_file("data/processed/combined_new_labeled_sentences.csv", "data/processed/combined_new_labeled_sentences_count.csv")
print(f"Label counts: {counts}")
counts = preprocessor.count_unique_labels_in_file("data/processed/combined_sentence_types.csv", "data/processed/combined_sentence_types_count.csv")
print(f"Label counts: {counts}")
counts = preprocessor.count_unique_labels_in_file("data/processed/combined_labeled_sentences.csv", "data/processed/combined_labeled_sentences_count.csv")
print(f"Label counts: {counts}")