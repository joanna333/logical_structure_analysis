import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from transformers import AutoModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# 1. Setup and preprocessing
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = texts.reset_index(drop=True)
        
        # Create label encoder if not already encoded
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 2. Model Architecture
class BiLSTMAttentionBERT(nn.Module):
    def __init__(self, 
                 hidden_dim=256,
                 num_classes=22,  # Based on your label distribution
                 num_layers=2,    # Multiple LSTM layers
                 dropout=0.5):    
        super().__init__()
        
        # Load BioBERT instead of BERT
        self.bert_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
        bert_dim = self.bert_model.config.hidden_size  # Still 768 for BioBERT basee
        # Dropout for BERT outputs
        self.dropout_bert = nn.Dropout(dropout)
        # Multi-layer BiLSTM
        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        
        # Regularization layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout + 0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = self.dropout_bert(bert_output.last_hidden_state)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            need_weights=False
        )
        
        # Pooling and normalization
        pooled = torch.mean(attn_out, dim=1)
        pooled = self.batch_norm(pooled)
        pooled = self.dropout2(pooled)
        
        # Classification
        return self.classifier(pooled)


# 3. Training Setup
def load_data():
    # Define file paths
    data_files = [
        'data/processed/all_labeled_sentences/combined_labeled_sentences.csv',
        'data/processed/all_labeled_sentences/combined_new_labeled_sentences.csv',
        'data/processed/all_labeled_sentences/combined_sentence_types.csv'
    ]
    
    # Load and combine all files
    dfs = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded {file}: {len(df)} rows")
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File {file} not found")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates if any
    combined_df = combined_df.drop_duplicates(subset=['Sentence'])
    
    print(f"Total combined dataset size: {len(combined_df)}")
    return combined_df

def create_data_loaders(df, batch_size=16, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = SentenceDataset(train_df['Sentence'], train_df['Label'], tokenizer)
    val_dataset = SentenceDataset(val_df['Sentence'], val_df['Label'], tokenizer)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_model(num_epochs=5, learning_rate=2e-5, weight_decay=0.01, device=None):
    """
    Training function for BiLSTM model with BERT embeddings.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    # Load and preprocess data
    df = load_data()  # Your existing load_data function
    train_loader, val_loader = create_data_loaders(df, tokenizer=tokenizer)  # You'll need to implement this
    
    # At the start of train_model
    label_encoder = LabelEncoder()
    labels = df['Label'].values  # Assuming df is your dataframe
    print(f"Labels: {labels}")
    label_encoder.fit(labels)
    
    # Initialize model with proper parameters
    model = BiLSTMAttentionBERT(
        hidden_dim=128,
        num_classes=len(label_encoder.classes_),  # You'll need to define label_encoder
        num_layers=2,
        dropout=0.5
    )
    
    # Set device if not provided
    if device is None:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and move to device
    model = model.to(device)
    
    # Initialize optimizer and loss
    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,  # L2 regularization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize training tracking
    best_val_acc = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    
   
    # Initialize tracking variables
    best_val_acc = 0
    no_improve_count = 0
    early_stopping_patience = 3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join('models', f'bilstm_bert_{timestamp}_valacc_{best_val_acc:.2f}.pt')
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # Validation phase
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # Print and log metrics
            print_and_handle_metrics(
                epoch=epoch,
                num_epochs=num_epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=best_val_acc,
                early_stopping_patience=early_stopping_patience
            )
            
            # Save best model and update counters
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_save_path = os.path.join('models', f'bilstm_bert_{timestamp}_valacc_{best_val_acc:.2f}.pt')
                save_best_model(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_acc=val_acc,
                    filename=model_save_path,
                    label_encoder=label_encoder
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Early stopping check
            if no_improve_count >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, None, None
        
    return model, train_loader, val_loader, best_val_acc  # Add best_val_acc to return values

# 4. Training function
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        try:
            outputs = model(input_ids, attention_mask)
        except RuntimeError as e:
            print(f"Error during forward pass: {e}")
            return None, None
        loss = criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, val_loader, criterion, device):
    """
    Evaluate model performance on validation set.
    
    Args:
        model: The PyTorch model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on (cuda/cpu)
    
    Returns:
        tuple: (average loss, accuracy percentage)
    """
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Store batch results
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * np.mean(np.array(predictions) == np.array(true_labels))
    avg_loss = total_loss / len(val_loader)
    
    print(classification_report(true_labels, predictions))

    
    return avg_loss, accuracy

# Add before main training loop
def save_checkpoint(state, filename):
    os.makedirs('models', exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def print_and_handle_metrics(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, best_val_acc, early_stopping_patience):
    """
    Print training metrics and return updated best validation accuracy.
    """
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    #print(f'Early Stopping Patience Remaining: {early_stopping_patience-no_improve_count}')

def save_best_model(model, optimizer, epoch, val_acc, filename, label_encoder):
    """
    Save the model checkpoint with relevant training information.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer used for training
        epoch: Current epoch number
        val_acc: Validation accuracy
        filename: Path where to save the model
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f"Saving model with {len(label_encoder.classes_)} classes")
    print("Classes:", label_encoder.classes_)
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    tokenizer.save_pretrained(os.path.join('models', 'tokenizer'))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'label_encoder_classes': label_encoder.classes_,
        'num_classes': len(label_encoder.classes_)
    }, filename, _use_new_zipfile_serialization=True)

def load_saved_model(model, optimizer, filename):
    """
    Load a saved model checkpoint.
    
    Args:
        model: The model architecture to load weights into
        optimizer: The optimizer to load state into
        filename: Path to the saved model checkpoint
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: The epoch where training stopped
        val_acc: The validation accuracy at save time
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, val_acc

# Modify main training loop
if __name__ == "__main__":
    model, train_loader, val_loader, best_val_acc = train_model()
    if model is not None:
        print(f"Training completed with best validation accuracy: {best_val_acc:.4f}")


