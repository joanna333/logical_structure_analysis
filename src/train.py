# src/train.py
import os
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from models.sentence_gnn import SentenceGNN
from data.dataset import SentenceRelationDataset
from data.dataset_preprocessor import move_invalid_files
import torch.nn.functional as F
from sklearn.metrics import classification_report
import logging
logging.basicConfig(level=logging.INFO)

def train(model, loader, test_loader, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        loss = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            print(f"Output shape: {out.shape}, Labels shape: {batch.y.shape}")  # Debug
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {loss/len(loader)}")
    # Add validation/testing steps as needed

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)  # Pass the entire batch object
            loss = F.cross_entropy(out, batch.y)  # Use appropriate loss function
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
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == "__main__":
    # Setup directories - fix the path construction
    data_root = 'data'
    raw_data_dir = os.path.join(data_root, 'processed', 'labeled_sentences')
    invalid_dir = os.path.join(data_root, 'processed', 'invalid_files')
    processed_dir = 'gnn_training_data'  # Remove extra 'processed' from path

    # Debug the paths
    logging.info(f"Raw data directory: {raw_data_dir}")
    logging.info(f"Processed directory: {os.path.join(data_root, 'processed', processed_dir)}")

    # Get and validate input files
    file_pattern = os.path.join(raw_data_dir, '*.txt')
    all_files = sorted(glob.glob(file_pattern))
    print(f"Found {len(all_files)} files")
    
    # Move invalid files and get valid ones
    valid_files = move_invalid_files(all_files, invalid_dir)
    print(f"Valid files: {len(valid_files)}")
    
    if not valid_files:
        print("No valid files found. Exiting.")
        exit(1)
    
    try:
        # First get the processed files list
        processed_files = []
        for file in valid_files:
            base_name = os.path.splitext(os.path.basename(file))[0]
            processed_path = os.path.join(data_root, 'processed', processed_dir, f"{base_name}.pt")
            processed_files.append(processed_path)

        # Create dataset with valid files
        dataset = SentenceRelationDataset(
            root=data_root,
            filenames=valid_files,
            processed_dir=processed_dir  # Ensure this matches the directory structure
        )
        
        # Add debug logging
        logging.debug(f"Dataset created with {len(dataset)} samples")
        logging.debug(f"First few processed files: {dataset.processed_file_paths[:3]}")
        
        # Verify files exist
        missing_files = [p for p in processed_files if not os.path.exists(p)]
        if missing_files:
            print(f"Missing {len(missing_files)} processed files:")
            for f in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {f}")
            raise FileNotFoundError("Missing processed files")
        
        # Calculate split indices
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        # Split dataset
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Before model creation
        logging.info(f"Creating model with {dataset.num_classes} classes")
        
        # Create model with correct number of output classes
        model = SentenceGNN(
            in_channels=768,     # BioBERT embedding dimension
            hidden_channels=512,
            num_classes=dataset.num_classes
        )
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        exit(1)
    print("Setup complete")
    train(model, train_loader, test_loader, num_epochs=100)