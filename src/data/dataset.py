import torch
from torch_geometric.data import Dataset, Data
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
import torch.serialization

class SentenceRelationDataset(Dataset):
    def __init__(self, root, filenames, transform=None):
        self.filenames = [os.path.abspath(f) for f in filenames]
        super().__init__(root, transform)
        
        # Device setup
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load models with optimizations
        model_name = "dmis-lab/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Process all files first
        self.process()
        
        # Verify all files were processed
        self._verify_processed_files()

    def _verify_processed_files(self):
        """Verify all processed files exist"""
        missing = []
        for idx in range(len(self.filenames)):
            path = os.path.join(self.processed_dir, f'data_{idx}.pt')
            if not os.path.exists(path):
                missing.append(idx)
        if missing:
            raise RuntimeError(f"Missing processed files for indices: {missing}")

    def _create_label_map(self):
        """Create label map from all possible relationships"""
        return {
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
        """Return list of processed file names"""
        return [f'data_{idx}.pt' for idx in range(len(self.filenames))]

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
        print("Processing files...")
        successful = 0
        
        for idx, filepath in enumerate(self.filenames):
            try:
                print(f"Processing {filepath}")
                
                # Skip if already processed
                output_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
                if os.path.exists(output_path):
                    successful += 1
                    continue
                    
                # Read CSV with proper error handling
                try:
                    df = pd.read_csv(filepath)
                    if df.empty:
                        print(f"Skipping empty file: {filepath}")
                        continue
                        
                    # Convert labels and create graph data
                    data = self._create_graph_data(df)
                    if data is not None:
                        torch.save(data, output_path)
                        successful += 1
                        
                except pd.errors.EmptyDataError:
                    print(f"Empty file: {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")
                    
            except Exception as e:
                print(f"Fatal error processing {filepath}: {str(e)}")
                
        print(f"Successfully processed {successful}/{len(self.filenames)} files")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No processed data found at {data_path}")
        
        # Add safe globals for PyTorch Geometric data types
        torch.serialization.add_safe_globals([
            'torch_geometric.data.data',
            'Data',
            'torch_geometric.data'
        ])
        
        return torch.load(data_path, weights_only=True)

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
        
if __name__ == "__main__":
    # Process one file from the directory
    dataset = SentenceRelationDataset('data/processed/labeled_sentences', ['data/processed/labeled_sentences/Acid_Secretion_sentences_with_labels.txt'])
    dataset.process()