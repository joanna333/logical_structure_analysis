import pytest
import torch
from unittest.mock import Mock, patch
import os
from datetime import datetime
from ..models.BiLSTM import train_model, BiLSTMAttentionBERT

# src/models/test_BiLSTM.py

import torch.nn as nn

class TestTrainModel:
    @pytest.fixture
    def mock_data(self):
        mock_loader = Mock()
        mock_loader.__iter__.return_value = [{
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.tensor([0, 1])
        }]
        mock_loader.__len__.return_value = 1
        return mock_loader

    @pytest.fixture
    def mock_bert(self):
        mock = Mock()
        mock.config.hidden_size = 768
        return mock

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        os.makedirs('models', exist_ok=True)
        yield
        # Cleanup test artifacts
        if os.path.exists('models/test_model.pt'):
            os.remove('models/test_model.pt')
        if os.path.exists('models'):
            os.rmdir('models')

    @patch('transformers.AutoModel.from_pretrained')
    @patch('pandas.read_csv')
    def test_train_model_saves_best_model(self, mock_read_csv, mock_auto_model, mock_data):
        # Setup
        mock_auto_model.return_value = self.mock_bert()
        device = torch.device('cpu')
        
        with patch('torch.save') as mock_save:
            model, _, _ = train_model(
                num_epochs=2,
                learning_rate=1e-5,
                device=device
            )
            
            # Verify model was saved
            mock_save.assert_called()
            
            # Verify model parameters
            assert isinstance(model, BiLSTMAttentionBERT)

    @patch('transformers.AutoModel.from_pretrained')
    def test_device_selection(self, mock_auto_model):
        mock_auto_model.return_value = self.mock_bert()
        
        with patch('torch.cuda.is_available', return_value=True):
            model, _, _ = train_model(num_epochs=1)
            assert str(next(model.parameters()).device) == 'cuda:0'

    @patch('transformers.AutoModel.from_pretrained')
    def test_error_handling(self, mock_auto_model):
        mock_auto_model.side_effect = Exception("BERT model loading failed")
        
        with pytest.raises(Exception):
            train_model(num_epochs=1)

    @patch('transformers.AutoModel.from_pretrained')
    def test_early_stopping(self, mock_auto_model, mock_data):
        mock_auto_model.return_value = self.mock_bert()
        
        # Mock validation scores that should trigger early stopping
        val_scores = [0.9, 0.89, 0.88, 0.87]  # Decreasing scores
        
        with patch('your_module.evaluate', side_effect=val_scores):
            model, _, _ = train_model(
                num_epochs=10,
                early_stopping_patience=3
            )
            # Verify training stopped early
            # Add assertions based on your early stopping implementation

def save_best_model(model, optimizer, epoch, val_acc, filename):
    """Add this function to BiLSTM.py"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }, filename)