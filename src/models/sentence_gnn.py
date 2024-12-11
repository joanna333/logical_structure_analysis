# src/models/sentence_gnn.py
import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling, SAGPooling, GraphConv
import logging
from torch_geometric.nn import SAGEConv, global_mean_pool, TransformerConv, GlobalAttention, GINConv

class ImprovedSentenceGNN(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_classes=32, 
                num_heads=8, dropout=0.1, pool_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.pool_ratio = pool_ratio
        
        # Input projection with same dimensionality
        self.input_proj = Linear(in_channels, hidden_channels)
        
        # Layer normalization
        self.layer_norms = torch.nn.ModuleList([
            LayerNorm(hidden_channels) for _ in range(3)
        ])
        
        # Calculate head dimensions to maintain total hidden size
        self.head_dim = hidden_channels // self.num_heads
        
        # Multi-layer attention with proper dimensions
        self.attention_layers = torch.nn.ModuleList([
            GATv2Conv(
                hidden_channels,
                self.head_dim,
                heads=self.num_heads,
                dropout=self.dropout,
                add_self_loops=True,
                concat=True  # Concatenate heads to maintain dimension
            )
            for _ in range(3)
        ])
        
        # Jump connection to combine intermediate representations
        self.jump = JumpingKnowledge(mode='cat')
        
        # Output layers
        self.out_norm = LayerNorm(hidden_channels * 3)  # For concatenated jump connections
        self.out_dropout = torch.nn.Dropout(dropout)
        self.out_proj = Linear(hidden_channels * 3, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Initial projection
        x = self.input_proj(x)
        
        # Store intermediate representations for jump connections
        jump_inputs = []
        
        # Process through attention layers
        for i, (attention, norm) in enumerate(zip(self.attention_layers, self.layer_norms)):
            # Apply attention
            attention_out = attention(x, edge_index)
            
            # Residual connection and normalization
            x = norm(attention_out + x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Store for jump connection
            jump_inputs.append(x)
        
        # Combine features from all layers
        x = self.jump(jump_inputs)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output projection
        x = self.out_norm(x)
        x = self.out_dropout(x)
        x = self.out_proj(x)
        
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Linear(channels, channels)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv2 = torch.nn.Linear(channels, channels)
        self.bn2 = torch.nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, **kwargs):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch.batch)
        return self.classifier(x)

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, **kwargs):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.attention = GlobalAttention(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 1)
            )
        )
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.attention(x, batch.batch)
        return self.classifier(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, **kwargs):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_add_pool(x, batch.batch)
        return self.classifier(x)

