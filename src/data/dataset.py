import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class TextDataset(Dataset):
    """A PyTorch Dataset class for processing text data for transformers.
    
    Inherits from torch.utils.data.Dataset to enable data loading with DataLoader.
    """
    
    def __init__(self, texts: list, tokenizer_name: str = 'bert-base-uncased'):
        """Initialize the dataset with texts and tokenizer.
        
        Args:
            texts (list): List of text strings to process
            tokenizer_name (str): Name of the pretrained tokenizer to use
        """
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.
        
        Returns:
            int: Number of text samples
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """Get a tokenized text sample at the specified index.
        
        Args:
            idx (int): Index of the text sample to retrieve
            
        Returns:
            dict: Dictionary containing tokenized inputs with keys:
                - input_ids: Tensor of token ids
                - attention_mask: Tensor indicating valid tokens
        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',    # Pad all samples to max_length
            truncation=True,         # Truncate texts longer than max_length
            max_length=512,          # Maximum sequence length
            return_tensors='pt'      # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }