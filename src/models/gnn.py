import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Add dimension reduction
        self.dim_reduction = torch.nn.Linear(input_dim, hidden_dim)
        # Update GAT layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1)
        
    def forward(self, x, edge_index, batch=None):
        # Reduce dimensionality first
        x = self.dim_reduction(x)
        x = F.relu(x)
        
        # GAT layers
        x, attention1 = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        x, attention2 = self.gat2(x, edge_index, return_attention_weights=True)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return F.log_softmax(x, dim=-1), (attention1, attention2)

def create_test_graph():
    # Simulate sentence embeddings (dim=768 for BERT-base)
    embedding_dim = 768
    num_nodes = 4
    
    # Create random embeddings for 4 sample sentences
    sentences = {
        0: "The heart pumps blood through arteries.",
        1: "This circulation provides oxygen to tissues.",
        2: "When oxygen levels drop, cells cannot function.",
        3: "Therefore, heart function is essential for survival."
    }
    
    # Simulate BERT embeddings (normally you'd use a real encoder)
    x = torch.randn(num_nodes, embedding_dim)
    
    # Define edge connections representing logical relations
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],  # Source nodes
        [1, 0, 2, 1, 3, 2]   # Target nodes
    ], dtype=torch.long)
    
    # Define logical connection types
    connection_types = {
        'causes': torch.tensor([1.0, 0.0, 0.0, 0.0]),
        'follows_from': torch.tensor([0.0, 1.0, 0.0, 0.0]),
        'if_then': torch.tensor([0.0, 0.0, 1.0, 0.0]),
        'therefore': torch.tensor([0.0, 0.0, 0.0, 1.0])
    }
    
    # Create edge attributes based on logical connections
    edge_attr = torch.stack([
        connection_types['causes'],      # 0->1: pumping causes circulation
        connection_types['follows_from'],# 1->0: circulation follows from pumping
        connection_types['if_then'],     # 1->2: if circulation fails, then cells suffer
        connection_types['causes'],      # 2->1: oxygen drop causes dysfunction
        connection_types['therefore'],   # 2->3: logical conclusion
        connection_types['if_then']      # 3->2: if heart fails, then cells suffer
    ])
    
    # Create graph data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes
    )
    
    # Store original sentences and connections for interpretation
    data.sentences = sentences
    data.connections = [
        ('causes', 0, 1),
        ('follows_from', 1, 0),
        ('if_then', 1, 2),
        ('causes', 2, 1),
        ('therefore', 2, 3),
        ('if_then', 3, 2)
    ]
    
    return data

def visualize_attention(attention_tuple, num_nodes):
    edge_index, att_weights = attention_tuple
    attention_matrix = torch.zeros((num_nodes, num_nodes))
    
    # Handle multi-head attention if present
    if len(att_weights.shape) > 1:
        att_weights = att_weights.mean(dim=1)  # Average over heads
    
    # Fill attention matrix
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        attention_matrix[src, dst] = att_weights[i]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix.detach().numpy(), annot=True, cmap='YlOrRd')
    plt.title('Attention Weights (Averaged over heads)')
    plt.xlabel('Target Node')
    plt.ylabel('Source Node')
    plt.show()

def visualize_graph(data):
    """Visualize graph structure using networkx"""
    # Convert to networkx
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    
    # Add nodes with features
    for i in range(len(data.x)):
        G.add_node(i, features=data.x[i].numpy())
    
    # Add edges
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    # Set up plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Add node labels with features
    labels = {i: f'Node {i}\n{data.x[i].numpy()}' for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Test Graph Visualization")
    plt.axis('off')
    plt.show()

def train_and_evaluate():
    graph_data = create_test_graph()
    print(f"Input feature dimensions: {graph_data.x.shape}")
    
    model = GATModel(
        input_dim=768,  # Match BERT embedding dimension
        hidden_dim=64,  # Increase hidden dimension for better representation
        output_dim=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create fake labels for demonstration
    node_labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    
    model.train()
    losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        out, (attention1, attention2) = model(graph_data.x, graph_data.edge_index)
        loss = F.nll_loss(out, node_labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:>3d}, Loss: {loss.item():.4f}')
    
    # Add debug prints
    print("\nDebugging model output:")
    out, attention_tuple = model(graph_data.x, graph_data.edge_index)
    print("Type of attention_tuple:", type(attention_tuple))
    print("Contents of attention_tuple:", attention_tuple)
    
    # Visualization after training
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(graph_data.x, graph_data.edge_index)
        print("\nFinal node predictions:", torch.exp(predictions))
        
        # Debug attention weights structure
        print("\nAttention weights type:", type(attention_weights))
        print("Attention weights structure:", attention_weights)
        
        # Correctly unpack nested tuples
        (edge_index1, att_weights1), (edge_index2, att_weights2) = attention_weights
        
        print("\nFirst layer attention:")
        print("Edge index shape:", edge_index1.shape)
        print("Weights shape:", att_weights1.shape)
        
        # Visualize first layer attention
        print("\nVisualizing first layer attention weights:")
        visualize_attention((edge_index1, att_weights1), num_nodes=4)

if __name__ == "__main__":
    train_and_evaluate()
    data = create_test_graph()
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Node feature dimension: {data.x.size(1)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Edge feature dimension: {data.edge_attr.size(1)}")
    print("\nLogical connections:")
    for conn, src, dst in data.connections:
        print(f"{data.sentences[src]} --[{conn}]--> {data.sentences[dst]}")