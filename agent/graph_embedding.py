#!./venv/bin/python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool

class GraphEmbeddingModel(nn.Module):
    """
    Graph Embedding Network that combines:
    - Node-level local features
    - Edge features from connected neighbors
    
    Args:
        node_in_dim (int): Dimension of input node features
        edge_in_dim (int): Dimension of input edge features
        hidden_dim (int): Hidden dimension size (default: 256)
        output_dim (int): Output embedding dimension (default: 32)
        num_layers (int): Number of GNN message passing layers (default: 3)
        num_heads (int): Number of attention heads (default: 4)
    """
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 32,
        num_layers: int = 3,
        dropout = 0.01,
    ):
        super().__init__()
        
        # Dimension configurations
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 1. Input projection layers
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)
        
        # 2. Local information encoder (message passing layers)
        self.local_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Use GINE to incorporate edge features
            self.local_layers.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    eps=0.1
                )
            )
        
        # 4. Output layer
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Performance prediction head
        self.performance_predictor = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Predict performance score
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Forward pass of the local graph embedding model.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_in_dim]
            edge_index (torch.Tensor): Edge indices [2, num_edges]
            edge_attr (torch.Tensor): Edge features [num_edges, edge_in_dim]
            
        Returns:
            torch.Tensor: Node embeddings [num_nodes, output_dim]
        """
        # 1. Feature projection
        h_node = F.relu(self.node_proj(x))
        h_edge = F.relu(self.edge_proj(edge_attr))
        
        # 2. Local neighborhood information aggregation
        h_local = h_node.clone()
        for conv in self.local_layers:
            h_local = F.relu(conv(h_local, edge_index, h_edge))
            h_local = F.dropout(h_local, p=0.1, training=self.training)
        
        h_out = self.output_norm(h_local)
        h_out = F.relu(self.output_proj(h_out))
        
        graph_emb = global_mean_pool(h_out, batch)

        performance = self.performance_predictor(graph_emb).squeeze(-1)
        
        return h_out, performance
    
    