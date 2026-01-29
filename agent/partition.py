import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch.nn.functional as F


class PartitionNode:
    """Partition node class corresponding to vertex V in the graph"""
    def __init__(self, p_id: int, isolation_level: int, mu: int, key_range_start: int, capacity: int = 8192):
        self.p_id = p_id # macro partition id
        self.isolation_level = isolation_level  # 0:SER, 1:SI, 2:RC
        self.mu = mu  # parameter for timestamp interval adjustment
        self.workload_intensity = 0.0
        self.key_range_start = key_range_start
        self.capacity = capacity
        if capacity % 8 != 0 or key_range_start % capacity != 0:
            raise ValueError("Capacity [" + str(capacity) + "] must be multiple of 8 and key_range_start must be aligned with capacity")
        self.level_from_top = 1
        self.is_leaf = True
        self.can_merge = False
        self.previous_embedding = None  # store previous embedding vector
        self.current_embedding = None   # store current embedding vector
        
        self.father = None  # type: Optional[PartitionNode]
        self.left = None  # type: Optional[PartitionNode]
        self.right = None # type: Optional[PartitionNode]
        # Each partition contains 8 micro partitions, each with 4 feature dimensions
        self.max_micro_partitions = 8
        self.capacity_per_micro_partition = capacity / self.max_micro_partitions
        self.p_start = self.key_range_start % (self.max_micro_partitions * self.capacity_per_micro_partition) // self.capacity_per_micro_partition;
        self.p_end = self.p_start + self.max_micro_partitions // np.power(2, self.level_from_top - 1)
        self.micro_partition_features = torch.randn(self.max_micro_partitions, 4)  # [write/read ratio, w_contention, r_contention, anomaly count]

    def get_node_features(self) -> torch.Tensor:
        """Get node feature vector by aggregating micro-partition features"""
        # concat each micro-partition feature to form node feature
        return self.micro_partition_features.view(-1)  # shape: [32]
    
    def is_includess_micro_partition(self, micro_idx: int) -> bool:
        return self.p_start <= micro_idx < self.p_end
    
    def update_micro_partition_features(self, micro_idx: int, features: List[float]):
        """
        micro idx: the offset of micro-partition in the partition
        Update features of a specific micro-partition
        """
        if micro_idx > self.max_micro_partitions or len(features) != 5:
            raise ValueError("Invalid micro-partition index or feature length")
        self.micro_partition_features[micro_idx] = torch.tensor(features[:4])
        self.workload_intensity += features[4]
        if self.is_leaf:
            return
        
        # recursively update the features
        if (self.left is not None) and self.left.is_includess_micro_partition(micro_idx):
            self.left.update_micro_partition_features(micro_idx, features)
        elif (self.right is not None) and self.right.is_includess_micro_partition(micro_idx):
            self.right.update_micro_partition_features(micro_idx, features)
    
    def update_embedding(self, embedding: torch.Tensor):
        """Update the current embedding and store the previous one"""
        self.previous_embedding = self.current_embedding
        self.current_embedding = embedding
        
    def get_workload_intensity(self) -> float:
        """Calculate overall workload intensity of the partition by average"""
        p_cnt = self.max_micro_partitions / np.power(2, self.level_from_top - 1)
        return self.workload_intensity / p_cnt
    
    def find_leaf_partition(self, micro_idx: int):
        """Find the leaf partition node that includes the specified micro-partition index"""
        if self.is_leaf:
            if self.is_includess_micro_partition(micro_idx):
                return self
            else:
                print("Error: micro-partition index not found in leaf partition")
                return None
        if (self.left is not None) and self.left.is_includess_micro_partition(micro_idx):
            return self.left.find_leaf_partition(micro_idx)
        elif (self.right is not None) and self.right.is_includess_micro_partition(micro_idx):
            return self.right.find_leaf_partition(micro_idx)
        else:
            return None

    def split(self):
        p_cnt = self.max_micro_partitions / np.power(2, self.level_from_top - 1)
        if p_cnt == 1:  # leaf node cannot split further
            return
                
        left = PartitionNode(self.p_id * 2, self.isolation_level, self.mu, self.key_range_start, self.capcity // 2)
        left.father = self
        left.level_from_top = self.level_from_top + 1
        left.micro_partition_features = torch.zeros(self.max_micro_partitions, 4)
        for i in range(int(p_cnt // 2)):
            left.micro_partition_features[int(self.p_start + i)] = self.micro_partition_features[int(self.p_start + i)]
        
        right = PartitionNode(self.p_id * 2 + 1, self.isolation_level, self.mu, self.key_range_start + self.capcity // 2, self.capcity // 2)
        right.father = self
        right.level_from_top = self.level_from_top + 1
        right.micro_partition_features = torch.zeros(self.max_micro_partitions, 4)
        for i in range(int(p_cnt // 2)):
            right.micro_partition_features[int(self.p_start + i + p_cnt // 2)] = self.micro_partition_features[int(self.p_start + i + p_cnt // 2)]
        
        # Update current node
        self.left = left
        self.right = right
        self.is_leaf = False
        self.can_merge = True
        
        # Update father level
        if self.father is not None:
            self.father.can_merge = False        


class PartitionGraph:
    """Partition graph structure representing relationships between partitions"""
    def __init__(self):
        self.nodes: Dict[int, PartitionNode] = {}
        self.edges: Dict[Tuple[int, int], int] = {}  # Edge weights: distributed transaction counts
        
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