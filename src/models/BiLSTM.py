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
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from nlpaug.augmenter.word import BackTranslationAug, RandomWordAug, ContextualWordEmbsAug
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf  
import signal
from contextlib import contextmanager
import wandb
import matplotlib.pyplot as plt

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
    def __init__(self, hidden_dim=256, num_classes=22, num_layers=2, dropout=0.5):
        super().__init__()
        
        # 1. Improved Architecture
        self.bert_model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2')

        self.dropout_bert = nn.Dropout(dropout)  # Added this line
        
        # Freeze initial BERT layers
        for param in list(self.bert_model.parameters())[:-2]:
            param.requires_grad = False
            
        # 2. Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        # 3. Enhanced Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,  # Increased heads
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Improved Regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout + 0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # 5. Deeper Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT encoding with dropout
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        sequence_output = self.dropout_bert(bert_output.last_hidden_state)
        
        # BiLSTM with layer norm
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.layer_norm1(lstm_out)  # First layer norm
        
        # Self-attention with dropout
        attn_out, _ = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            need_weights=False
        )
        attn_out = self.dropout1(attn_out)  # First dropout
        attn_out = self.layer_norm2(attn_out)  # Second layer norm
        
        # Pooling and normalization
        pooled = torch.mean(attn_out, dim=1)
        pooled = self.batch_norm(pooled)  # Batch norm
        pooled = self.dropout2(pooled)  # Second dropout
        
        # Classification
        return self.classifier(pooled)


# 3. Training Setup
def load_data():
    # Define file paths
    data_files = [
        'data/processed/all_labeled_sentences/combined_labeled_sentences.csv',
        'data/processed/all_labeled_sentences/combined_new_labeled_sentences.csv',
        'data/processed/all_labeled_sentences/combined_sentence_types.csv',
        'data/processed/all_labeled_sentences/combined_ai_data.csv'
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

def train_model(num_epochs=1, learning_rate=2e-5, weight_decay=0.01, device=None):
    """Training function for BiLSTM model with BERT embeddings."""
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    df = load_data()
    
    # Data augmentation
    # augmented_df = apply_augmentation(df)
    # train_loader, val_loader = create_data_loaders(augmented_df, tokenizer=tokenizer)
    
    train_loader, val_loader = create_data_loaders(df, tokenizer=tokenizer)

    # Label encoding
    label_encoder = LabelEncoder()
    labels = df['Label'].values
    label_encoder.fit(labels)
    
    # Model initialization
    model = BiLSTMAttentionBERT(
        hidden_dim=256,
        num_classes=len(label_encoder.classes_),
        num_layers=2,
        dropout=0.5
    )
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Training tracking
    best_val_acc = 0
    grad_accumulation_steps = 2
    wandb.init(
    project="bilstm-classification",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epochs
        }
    )

    wandb.config.update({
        
        # Dataset Stats
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "num_classes": len(label_encoder.classes_),
        
        # Training Config
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__ if scheduler else "none",
    })

    # At the start of training
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            # Gradient clipping
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accumulation_steps

            # Inside training loop, after computing loss
            train_pred = torch.argmax(outputs, dim=1)
            train_acc = (train_pred == labels).float().mean()
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_acc=val_acc,
                filename=f'models/model_epoch{epoch}_acc{val_acc:.2f}.pt',
                label_encoder=label_encoder
            )
        
        # After each epoch
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print epoch stats
        print_and_handle_metrics(
            epoch=epoch,
            num_epochs=num_epochs,
            train_loss=total_loss/len(train_loader),
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            best_val_acc=best_val_acc,
            early_stopping_patience=early_stopping.patience
        )
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "loss_gap": train_loss - val_loss,
            "accuracy_gap": train_acc - val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }, step=epoch)

        train_accuracies = train_accuracies.cpu() 
        val_accuracies = val_accuracies.cpu()
        train_losses = train_losses.cpu()
        val_losses = val_losses.cpu()
        # Plot learning curves
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title('Loss Curves')

        plt.subplot(1,2,2)
        plt.plot(train_accuracies, label='Train Acc')
        plt.plot(val_accuracies, label='Val Acc')
        plt.legend()
        plt.title('Accuracy Curves')
        wandb.log({"learning_curves": wandb.Image(plt)})

    wandb.finish()
    return model, train_loader, val_loader, best_val_acc

def apply_augmentation(df, aug_factor=0.3, timeout=30):
    """
    Apply text augmentation with timeout control
    """
    augmented_sentences = []
    augmented_labels = []
    
    # Initialize augmenters
    back_trans_aug = BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en'
    )
    
    synonym_aug = naw.SynonymAug(aug_p=0.3)
    
    n_samples = int(len(df) * aug_factor)
    aug_indices = np.random.choice(len(df), n_samples, replace=False)
    
    for idx in aug_indices:
        try:
            with time_limit(timeout):
                text = df['Sentence'].iloc[idx]
                label = df['Label'].iloc[idx]
                
                # Controlled augmentation
                augmented_text = synonym_aug.augment(text)[0]
                augmented_sentences.append(augmented_text)
                augmented_labels.append(label)
                
        except TimeoutException:
            print(f"Augmentation timed out for index {idx}")
            continue
        except Exception as e:
            print(f"Error at index {idx}: {str(e)}")
            continue
    
    aug_df = pd.DataFrame({
        'Sentence': augmented_sentences,
        'Label': augmented_labels
    })
    
    return pd.concat([df, aug_df], ignore_index=True)

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


