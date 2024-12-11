import csv
import logging
import torch
from torch_geometric.data import Dataset, Data
from label_mapping import LABEL_MAP
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch.serialization
#from .dataset_preprocessor import validate_file, move_invalid_files
import glob
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    # Load label map from external config for better maintainability


class SentenceRelationDataset(Dataset):
    """Dataset for sentence relations in documents.
    
    Handles processing and loading of sentence relation data for graph-based analysis.
    
    Attributes:
        LABEL_MAP (Dict[str, int]): Mapping of relation types to numeric indices
    """
    


    @classmethod
    def get_label_name(cls, label_id: int) -> str:
        """Get relation name from numeric label ID."""
        inv_map = {v: k for k, v in cls.LABEL_MAP.items()}
        return inv_map.get(label_id, "Unknown")

    @classmethod
    def validate_label(cls, label: str) -> bool:
        """Validate if a label exists in the mapping."""
        return label in cls.LABEL_MAP

    def __init__(self, root: str, filenames: List[str], transform=None, pre_transform=None):
        """Initialize the dataset.
        
        Args:
            root: Root directory for processed data
            filenames: List of input CSV files to process
            transform: Transform to apply to processed data
            pre_transform: Transform to apply before processing
        """
        # Validate input files
        self.filenames = []
        for f in filenames:
            if os.path.exists(f) and f.endswith('.csv'):
                logger.info(f"Adding file: {f}")
                self.filenames.append(os.path.abspath(f))
            else:
                logger.warning(f"Skipping invalid file: {f}")
        
        if not self.filenames:
            logger.error("No valid input files provided")
            raise ValueError("No valid input files provided")
            
        # Set up processing directory
        self.processed_dir_name = 'gnn_training_data'
        super().__init__(root, transform, pre_transform)
        self.label_map = self.LABEL_MAP
        
        # Device setup with fallback
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using MPS device")
        else:
            self.device = "cpu"
            logger.info("Using CPU device")
        
        # Load models with error handling
        model_name = "dmis-lab/biobert-v1.1"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info("Successfully loaded BioBERT model")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
        
        # Create and validate processed directory
        try:
            os.makedirs(self.processed_dir, exist_ok=True)
            logger.info(f"Using processed directory: {self.processed_dir}")
        except Exception as e:
            logger.error(f"Failed to create processed directory: {e}")
            raise
        
        # Process files and track status
        logger.info(f"Processing {len(self.filenames)} files...")
        self.process()
        
        # Create list of processed file paths
        self.processed_file_paths = []
        processed_count = 0
        for file in self.filenames:
            base_name = os.path.splitext(os.path.basename(file))[0]
            processed_path = os.path.join(self.processed_dir, f"{base_name}.pt")
            if os.path.exists(processed_path):
                self.processed_file_paths.append(processed_path)
                processed_count += 1
                
        logger.info(f"Successfully processed {processed_count}/{len(self.filenames)} files")
        
        if not self.processed_file_paths:
            logger.error("No data was processed from any file")
            raise RuntimeError("No data was processed")

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
        idx = 0
        for filepath in self.raw_paths:
            data_list = self._process_file(filepath)
            for data in data_list:
                torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    def __len__(self):
        return self.len()
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # Validate edge_index format
        if data.edge_index.shape[0] != 2:
            data.edge_index = data.edge_index.t().contiguous()
        
        # Ensure correct dtype
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.to(torch.float)
        
        return data

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

    def _process_file(self, filepath):
        data_list = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            try:
                sentence, label = line.strip().split('\t')
                words = sentence.split()
                if not words:
                    continue

                tokens = self.tokenizer(words, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.embedding_model(**tokens)
                    embeddings = outputs.last_hidden_state.squeeze(0)

                x = embeddings
                edge_index = []
                num_words = len(words)
                for i in range(num_words - 1):
                    edge_index.append([i, i + 1])
                    edge_index.append([i + 1, i])
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                y = torch.tensor([int(label)], dtype=torch.long)

                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)

            except ValueError as e:
                logging.warning(f"Error processing line: {line.strip()} - {e}")
                continue
            except Exception as e:
                logging.error(f"Unexpected error processing line: {line.strip()} - {e}")
                continue

        return data_list

def save_processed_data(data, path):
    try:
        torch.save(data, path)
        logging.info(f"Successfully saved data to {path}")
    except (IOError, OSError) as e:
        logging.error(f"File I/O error while saving to {path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error saving to {path}: {str(e)}")
        raise

def process_file(file_path):
    data_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            reader = csv.reader(f)
            
            for row in reader:
                try:
                    if len(row) != 2:
                        logging.warning(f"Skipping malformed row: {row}")
                        continue
                        
                    sentence, label = row
                    sentence = sentence.strip('"')
                    label = label.strip()
                    
                    # Validate label
                    if label not in LABEL_MAP:
                        logging.warning(f"Unknown label '{label}' in row: {row}")
                        continue
                    
                    # Create graph data
                    words = sentence.split()
                    x = torch.randn((len(words), 768))
                    
                    edge_index = []
                    for i in range(len(words) - 1):
                        edge_index.append([i, i + 1])
                        edge_index.append([i + 1, i])
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    
                    # Use LABEL_MAP for label encoding
                    y = torch.tensor([LABEL_MAP[label]], dtype=torch.long)
                    
                    data = Data(x=x, edge_index=edge_index, y=y)
                    data_list.append(data)
                    
                except (ValueError, KeyError) as e:
                    logging.warning(f"Error processing row: {row} - {e}")
                    continue
                    
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise
        
    return data_list

def save_individual_graphs(data_list, output_dir):
    """Save each graph as a separate file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, data in enumerate(data_list):
            output_path = os.path.join(output_dir, f'graph_{idx}.pt')
            try:
                torch.save(data, output_path)
                #logging.info(f"Saved graph {idx} to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save graph {idx}: {e}")
                continue
                
        logging.info(f"Successfully saved {len(data_list)} graphs to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create/access output directory: {e}")
        raise

def process_directory(input_dir):
    """Process all .csv files in directory and return combined data list."""
    all_data = []
    input_dir = Path(input_dir)
    
    # Get all txt files
    input_csv_files = list(input_dir.glob('*.csv'))
    logging.info(f"Found {len(input_csv_files)} files to process")
    
    for file_path in input_csv_files:
        try:
            logging.info(f"Processing file: {file_path}")
            file_data = process_file(file_path)
            if file_data:
                all_data.extend(file_data)
                logging.info(f"Added {len(file_data)} graphs from {file_path}")
            else:
                logging.warning(f"No data processed from {file_path}")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue
            
    return all_data


def process_sentence(sentence):
    """Convert a single sentence to graph format for prediction."""
    try:
        # Preprocess sentence
        sentence = sentence.strip('"')
        words = sentence.split()
        
        # Create feature matrix (same dimensions as training data)
        x = torch.randn((len(words), 768))  # BERT embedding size
        
        # Create edge indices for sequential connections
        edge_index = []
        for i in range(len(words) - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create Data object (without label for prediction)
        data = Data(x=x, edge_index=edge_index)
        
        return data
        
    except Exception as e:
        logging.error(f"Error processing sentence: {sentence} - {e}")
        raise

def process_graph_data(edge_index, node_features):
    """Process graph data to ensure correct format"""
    # Convert edge_index to proper format
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Ensure edge_index is 2xN
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    
    # Convert node features if needed
    if not isinstance(node_features, torch.Tensor):
        node_features = torch.tensor(node_features, dtype=torch.float)
        
    return edge_index, node_features

def validate_graph_data(edge_index, node_features):
    """Validate graph data structure"""
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Check shape
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Edge index must have shape [2, num_edges], got {edge_index.shape}")
    
    # Check dtype
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    
    # Check index range
    num_nodes = node_features.size(0)
    if torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes):
        raise ValueError(f"Edge indices must be in range [0, {num_nodes-1}]")
    
    # Check connectivity (no isolated nodes)
    connected_nodes = torch.unique(edge_index)
    if len(connected_nodes) < num_nodes:
        logging.warning(f"Graph has {num_nodes - len(connected_nodes)} isolated nodes")
    
    return edge_index



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_dir = 'data/processed/all_labeled_sentences'
    output_dir = 'data/processed/graphs'
    
    try:
        # Process all files in directory
        processed_data = process_directory(input_dir)
        
        if processed_data:
            # Extract graph data from processed sentences
            edge_indices = []
            node_features = []
            
            for data in processed_data:
                if hasattr(data, 'edge_index'):
                    edge_indices.append(data.edge_index)
                if hasattr(data, 'node_features'):
                    node_features.append(data.node_features)
            
            # Validate edge indices and node features
            for edge_index, node_feat in zip(edge_indices, node_features):
                edge_index = validate_graph_data(edge_index, node_feat)
                
            # Save the processed graphs    
            save_individual_graphs(processed_data, output_dir)
            logging.info(f"Successfully processed {len(processed_data)} total sentences")
        else:
            logging.error("No data was processed from any file")
            
    except Exception as e:
        logging.error(f"Failed to process dataset: {str(e)}")
        raise