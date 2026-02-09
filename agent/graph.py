import torch
from typing import List, Dict, Tuple, Optional
from torch_geometric.data import Data

from agent.partition import PartitionNode


class PartitionGraph:
    """Partition graph structure representing relationships between partitions"""
    def __init__(self, node_cnt: int):
        self.nodes: Dict[int, PartitionNode] = {}
        self.edges: Dict[Tuple[int, int], int] = {}  # Edge weights: distributed transaction counts
        self.tput: float
        self.abort_ratio: float
        
    def set_features(self, tput: float, abort_ratio: float):
        self.tput = tput
        self.abort_ratio = abort_ratio
        
    def get_features(self) -> torch.Tensor:
        """Get graph-level feature vector"""
        return torch.tensor([self.tput, self.abort_ratio])
        
    def add_partition(self, partition: PartitionNode):
        """Add partition node to graph"""
        self.nodes[partition.p_id] = partition
        
    def add_edge(self, partition_i: int, partition_j: int, transaction_count: int = 0):
        """Add undirected edge between partitions"""
        edge_key = tuple(sorted((partition_i, partition_j)))
        self.edges[edge_key] = transaction_count
        
    def get_adjacency_matrix(self) -> torch.Tensor:
        """Get normalized adjacency matrix"""
        n_nodes = len(self.nodes)
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        
        node_ids = sorted(self.nodes.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Build adjacency matrix with edge weights
        for (i, j), weight in self.edges.items():
            idx_i, idx_j = id_to_idx[i], id_to_idx[j]
            adj_matrix[idx_i, idx_j] = weight
            adj_matrix[idx_j, idx_i] = weight  # Undirected graph
            
        return adj_matrix
    
    def get_node_feature_matrix(self) -> torch.Tensor:
        """Get node feature matrix for all partitions"""
        node_ids = sorted(self.nodes.keys())
        features = []
        for node_id in node_ids:
            features.append(self.nodes[node_id].get_node_features())
        return torch.stack(features)
    
    def to_pyg_data(self) -> Data:
        node_ids = sorted(self.nodes.keys())
        id_map = {pid: i for i, pid in enumerate(node_ids)}

        # node features
        x = torch.stack([self.nodes[pid].get_node_features() for pid in node_ids])

        # edges
        edge_index = []
        edge_attr = []

        for (i, j), weight in self.edges.items():
            edge_index.append([id_map[i], id_map[j]])
            edge_index.append([id_map[j], id_map[i]])  # undirected

            # edge feature = transaction count
            edge_attr.append([weight])
            edge_attr.append([weight])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return Data(x=x, y=self.tput, edge_index=edge_index, edge_attr=edge_attr)