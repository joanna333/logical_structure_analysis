import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, texts: list, tokenizer_name: str = 'bert-base-uncased'):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }