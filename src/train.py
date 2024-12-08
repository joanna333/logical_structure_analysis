# src/train.py
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report
from data.dataset import SentenceRelationDataset
from models.sentence_gnn import SentenceGNN
import glob

def train(model, train_loader, test_loader, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_loss = float('inf')
    
    # Create directory for model checkpoints
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        val_loss = evaluate(model, test_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'models/best_model.pt')

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out, batch.y)
            total_loss += loss.item()
            
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    
    return total_loss / len(loader)

def predict(model, sentence, tokenizer, label_map):
    """Make prediction for a single sentence"""
    model.eval()
    with torch.no_grad():
        # Get embedding
        inputs = tokenizer(
            sentence, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(model.device)
        
        embedding = model.encoder(**inputs).last_hidden_state[:, 0, :]
        
        # Get prediction
        out = model(embedding)
        pred = out.argmax(dim=1)
        
        # Convert to label
        inv_map = {v: k for k, v in label_map.items()}
        return inv_map[pred.item()]

def load_model(model, path):
    """Load trained model"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    # Get all txt files from directory
    file_pattern = os.path.join('data', 'processed', 'labeled_sentences', '*.txt')
    filenames = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(filenames)} files")
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Load data with found files
    dataset = SentenceRelationDataset(
        root='data',  # Project root directory
        filenames=filenames  # All text files
    )
    
    # Calculate split indices
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model with correct number of output classes
    model = SentenceGNN(
        in_channels=768,     # BioBERT embedding dimension
        hidden_channels=64,  # Hidden dimension
        out_channels=len(dataset.label_map)  # Number of relationship types
    )
    print(model)
    
    train(model, train_loader, test_loader, num_epochs=1)