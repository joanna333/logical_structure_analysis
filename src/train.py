# src/train.py
import os
import glob
import torch
from torch_geometric.loader import DataLoader
from models.sentence_gnn import ImprovedSentenceGNN, GraphTransformer, GraphSAGE, GIN  
#from data.dataset import process_sentence
from data.dataset_preprocessor import move_invalid_files
from data.graph_dataset import GraphDataset
import torch.nn.functional as F
import logging
from torch.utils.data import random_split
logging.basicConfig(level=logging.INFO)

def debug_graph_batch(data):
    logging.info("Graph data debug info:")
    logging.info(f"Edge index shape: {data.edge_index.shape}")
    logging.info(f"Edge index dtype: {data.edge_index.dtype}")
    logging.info(f"Node features shape: {data.x.shape}")
    logging.info(f"Labels shape: {data.y.shape}")
    # Only log batch info if it exists
    if hasattr(data, 'batch') and data.batch is not None:
        logging.info(f"Batch info: {data.batch.shape}")

def train(model, train_loader, val_loader, config):
    try:
        for batch in train_loader:
            # Debug first batch
            debug_graph_batch(batch)
            
            # Ensure proper device and format
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            batch = batch.to(device)
            if batch.edge_index.shape[0] != 2:
                batch.edge_index = batch.edge_index.t().contiguous()
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    # In train function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y)
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader)
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)
        scheduler.step(val_acc)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in loader:
            # Match the calling pattern from training
            out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            loss = criterion(out, batch.y)
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# def predict(model, sentence, label_map):
#     """Make prediction on single sentence."""
#     embedding = process_sentence(sentence)
#     model.eval()
#     with torch.no_grad():
#         out = model(embedding)
#         pred = out.argmax(dim=1)
        
#         # Convert to label
#         inv_map = {v: k for k, v in label_map.items()}
#         return inv_map[pred.item()]

def load_model(model, path):
    """Load trained model"""
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def setup_training(data_root='data/processed/graphs', val_split=0.2):
    """Setup training with pre-processed graph files."""
    logging.info(f"Loading graphs from: {data_root}")
    
    try:
        # Create dataset from saved graphs
        dataset = GraphDataset(root=data_root)
        
        # Validate first graph to catch formatting issues early
        first_graph = dataset[0]
        debug_graph_batch(first_graph)
        
        # Ensure edge indices are properly formatted
        if first_graph.edge_index.shape[0] != 2:
            logging.warning("Reformatting edge indices...")
            dataset = [format_graph_data(graph) for graph in dataset]
        
        logging.info(f"Dataset size: {len(dataset)} graphs")
        
        # Create train/val split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logging.info(f"Split into {train_size} train and {val_size} validation samples")
        return train_dataset, val_dataset
        
    except Exception as e:
        logging.error(f"Failed to setup training: {str(e)}")
        raise
        
    except Exception as e:
        logging.error(f"Failed to setup training: {str(e)}")
        raise

def format_graph_data(data):
    """Ensure graph data is in correct format for PyTorch Geometric"""
    logging.debug(f"Formatting graph data: {data}")
    
    # Validate and convert node features
    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)
    
    # Convert edge indices to long dtype with strict checking
    if not isinstance(data.edge_index, torch.Tensor):
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.long)
    else:
        data.edge_index = data.edge_index.long()
    
    # Ensure edge_index is 2xN and values are valid
    if data.edge_index.dim() != 2:
        raise ValueError(f"Edge index must be 2-dimensional, got {data.edge_index.dim()} dimensions")
    
    if data.edge_index.size(0) != 2:
        data.edge_index = data.edge_index.t()
    
    # Validate edge indices
    num_nodes = data.x.size(0)
    if torch.any(data.edge_index < 0) or torch.any(data.edge_index >= num_nodes):
        raise ValueError(f"Edge indices must be in range [0, {num_nodes-1}]")
    
    # Make contiguous and ensure correct dtype
    data.edge_index = data.edge_index.contiguous()
    
    # Detailed debug logging
    logging.debug(f"Edge index dtype: {data.edge_index.dtype}")
    logging.debug(f"Edge index shape: {data.edge_index.shape}")
    logging.debug(f"Edge index min/max: {data.edge_index.min()}/{data.edge_index.max()}")
    logging.debug(f"Number of nodes: {num_nodes}")
    
    return data

def setup_config():
    return {
        # Training params
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        
        # Model architecture
        'hidden_channels': 128,
        'num_heads': 8,
        'dropout': 0.3,
        'pool_ratio': 0.8,
        
        # Model input/output
        'in_channels': 768,  # BERT embedding dimension
        'num_classes': 22     # Number of classes
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        config = setup_config()
        train_dataset, val_dataset = setup_training()
        
        # Create data loaders using config
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size']
        )
        
        # Initialize model using config
        model = ImprovedSentenceGNN(
            in_channels=config['in_channels'],
            hidden_channels=config['hidden_channels'],
            num_classes=config['num_classes'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            pool_ratio=config['pool_ratio']
        )
        
        # Train using config
        train(model, train_loader, val_loader, config)
        
    except Exception as e:
        logging.error(f"Failed to setup training: {e}")
        raise


