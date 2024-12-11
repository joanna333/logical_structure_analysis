import os
import glob
import logging
import torch
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.graphs = sorted(glob.glob(os.path.join(root, '*.pt')))
        
    def len(self):
        return len(self.graphs)
        
    def get(self, idx):
        return torch.load(self.graphs[idx])

def setup_training(data_root='data/processed/graphs'):
    """Setup training data from processed graph files."""
    logging.info(f"Loading graphs from: {data_root}")
    
    if not os.path.exists(data_root):
        raise ValueError(f"Graphs directory not found: {data_root}")
    
    try:
        # Create dataset from saved graphs
        dataset = GraphDataset(root=data_root)
        logging.info(f"Loaded dataset with {len(dataset)} graphs")
        
        return dataset
        
    except Exception as e:
        logging.error(f"Error loading graph dataset: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        dataset = setup_training()
        logging.info("Dataset loaded successfully")
        
    except Exception as e:
        logging.error(f"Failed to setup training: {e}")
        raise