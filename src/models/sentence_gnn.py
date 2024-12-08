# src/models/sentence_gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class SentenceGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(SentenceGNN, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, batch):
        # Extract node features, edge indices, and batch indices from the batch object
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch  # This is the tensor indicating each node's batch

        # First Graph Attention Layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # Second Graph Attention Layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Third Graph Attention Layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Global Mean Pooling
        x = global_mean_pool(x, batch_idx)

        # Dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)

        # Fully Connected Layer
        x = self.classifier(x)

        return x