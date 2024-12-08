import logging
import torch
from torch_geometric.data import Dataset, Data
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
import torch.serialization
from data.dataset_preprocessor import validate_file, move_invalid_files
import glob
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

class SentenceRelationDataset(Dataset):
    # Define label map as class attribute since it's static
    LABEL_MAP = {
        'Explanation': 0,
        'Addition': 1,
        'Causal': 2,
        'Emphasis': 3,
        'Summary': 4,
        'Conditional': 5,
        'Sequential': 6,
        'Comparison': 7,
        'Definition': 8,
        'Contrast': 9,
        'Elaboration': 10,
        'Illustration': 11,
        'Concession': 12,
        'Generalization': 13,
        'Inference': 14,
        'Problem Solution': 15,
        'Contrastive Emphasis': 16,
        'Purpose': 17,
        'Clarification': 18,
        'Enumeration': 19,
        'Cause and Effect': 20,
        'Temporal Sequence': 21
    }

    def __init__(self, root, filenames, processed_dir='gnn_training_data', transform=None):
        self.label_map = self.LABEL_MAP
        
        # Validate and filter files first
        invalid_dir = os.path.join(root, 'processed', 'invalid_files')
        valid_files = move_invalid_files(filenames, invalid_dir)
        
        if not valid_files:
            raise RuntimeError("No valid files found for processing")
            
        # Continue with valid files only
        self.processed_dir_name = processed_dir
        self.filenames = [os.path.abspath(f) for f in valid_files]
        super().__init__(root, transform)
        
        # Device setup
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load models with optimizations
        model_name = "dmis-lab/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Process all files first
        self.process()
        
        # Create list of processed file paths
        self.processed_file_paths = []
        for file in filenames:
            base_name = os.path.splitext(os.path.basename(file))[0]
            processed_path = os.path.join( self.processed_dir, f"{base_name}.pt")
            self.processed_file_paths.append(processed_path)

    @property 
    def num_classes(self):
        """Return number of unique classes"""
        return len(self.label_map)

    @property
    def processed_dir(self):
        """Override processed_dir property to use custom directory"""
        return os.path.join(self.root, 'processed', self.processed_dir_name)

    def _setup_device(self):
        """Setup compute device"""
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _load_model(self):
        """Load tokenizer and model with error handling"""
        try:
            model_name = "dmis-lab/biobert-v1.1"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            return tokenizer, model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError("Failed to initialize tokenizer and model")

    @property
    def raw_file_names(self):
        """Return list of raw file names"""
        return [os.path.basename(f) for f in self.filenames]

    @property
    def processed_file_names(self):
        """Return list of processed file names with original names"""
        return [f"{os.path.splitext(os.path.basename(f))[0]}.pt" for f in self.filenames]

    def _is_file_empty(self, filepath: str) -> bool:
        """Check if file is empty or contains only whitespace"""
        try:
            with open(filepath, 'r') as f:
                first_line = f.readline()
                if not first_line.strip():
                    return True
                return False
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return True

    def process(self):
        """Process raw data into graph format"""
        # Create processed directory if it doesn't exist 
        os.makedirs(self.processed_dir, exist_ok=True)
        print(f"Processing files into {self.processed_dir}...")
        successful = 0
        
        for idx, filepath in enumerate(self.filenames):
            try:
    #            print(f"Processing {filepath}")
                
                # Use full path for processing
                abs_filepath = os.path.abspath(filepath)
                
                # Get output path using original filename while preserving directory structure
                rel_path = os.path.relpath(abs_filepath, self.root)
                base_name = os.path.splitext(os.path.basename(abs_filepath))[0]
                output_path = os.path.join(self.processed_dir, f'{base_name}.pt')
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Skip if already processed
                if os.path.exists(output_path):
                    successful += 1
                    continue
                    
                # Validate file before processing
                is_valid, error = validate_file(abs_filepath)
                if not is_valid:
                    print(f"Skipping invalid file {abs_filepath}: {error}")
                    continue

                # Process valid file
                data = self._process_single_file(filepath, idx)
                if data is not None:
                    save_processed_data(data, output_path)
                    successful += 1
                    
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                
        print(f"Successfully processed {successful}/{len(self.filenames)} files")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def __getitem__(self, idx):
        try:
            file_path = self.processed_file_paths[idx]
            logging.debug(f"Loading data from {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Processed file not found: {file_path}")
                
            return torch.load(file_path)
            
        except Exception as e:
            logging.error(f"Error loading data at index {idx}: {str(e)}")
            raise

    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.filenames)

    def _get_embeddings(self, sentences, batch_size=32):
        """Get BERT embeddings for sentences"""
        with torch.no_grad():
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] token

    def _create_edges(self, num_nodes: int):
        """Create fully connected edges between nodes"""
        edges = []
        # Create bidirectional edges between all nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t()

    def _process_single_file(self, filepath, idx):
        """Process a single file into a graph data object"""
        try:
            # Read CSV with proper header
            df = pd.read_csv(filepath)
            
            if df.empty:
                print(f"Skipping empty file: {filepath}")
                return None
                
            # Get sentences and labels
            sentences = df['Sentence'].tolist()
            labels = df['Label'].tolist()
            
            # Convert labels to numeric using label_map
            numeric_labels = [self.label_map.get(label, 0) for label in labels]
            
            # Get embeddings
            embeddings = self._get_embeddings(sentences)
            
            # Create edge index (fully connected graph)
            edge_index = self._create_edges(len(sentences))
            
            # Create graph data object
            data = Data(
                x=embeddings,
                edge_index=edge_index,
                y=torch.tensor(numeric_labels, dtype=torch.long)
            )
            
            return data
            
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")
            return None
        
def save_processed_data(data, path):
    try:
        torch.save(data, path)
    except Exception as e:
        logging.error(f"Error saving to {path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Process one file from the directory
    dataset = SentenceRelationDataset('data/processed/labeled_sentences', ['data/processed/labeled_sentences/Acid_Secretion_sentences_with_labels.txt'])
    dataset.process()