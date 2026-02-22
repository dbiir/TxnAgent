import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class PartitionNode:
    """Partition node class corresponding to vertex V in the graph"""
    _next_id = 0  # class-level global unique ID counter

    @classmethod
    def allocate_id(cls) -> int:
        """Allocate the next globally unique partition ID."""
        pid = cls._next_id
        cls._next_id += 1
        return pid

    @classmethod
    def reset_id_counter(cls, start: int = 0):
        """Reset the global ID counter (e.g. at init time)."""
        cls._next_id = start

    def __init__(self, p_id: int, isolation_level: int, mu: int, key_range_start: int, capacity: int = 8192):
        self.p_id = p_id # macro partition id
        # Track the max ID we've seen so the counter stays ahead
        if p_id >= PartitionNode._next_id:
            PartitionNode._next_id = p_id + 1
        self.isolation_level = isolation_level  # 0:SER, 1:SI, 2:RC
        self.mu = mu  # parameter for timestamp interval adjustment
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
        self.micro_partition_features = torch.randn(self.max_micro_partitions, 4)  # [write/read ratio, abort_rate, workload_intensive, isolation_level]

    def get_node_features(self) -> torch.Tensor:
        """Get node feature vector by aggregating micro-partition features"""
        # concat each micro-partition feature to form node feature
        return self.micro_partition_features.view(-1)  # shape: [32]

    @property
    def workload_intensity(self) -> float:
        """Average workload intensity across active micro-partitions (feature index 2)."""
        start = int(self.p_start)
        end = int(self.p_end)
        return self.micro_partition_features[start:end, 2].max().item()
    
    def is_includess_micro_partition(self, micro_idx: int) -> bool:
        return self.p_start <= micro_idx < self.p_end
    
    def update_micro_partition_features(self, micro_idx: int, features: List[float]):
        """
        micro idx: the offset of micro-partition in the partition
        Update features of a specific micro-partition
        """
        if micro_idx > self.max_micro_partitions or len(features) != 4:
            raise ValueError("Invalid micro-partition index or feature length")
        self.micro_partition_features[micro_idx] = torch.tensor(features)
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

    def split(self, iso_l, mu_l, iso_r, mu_r):
        p_cnt = self.max_micro_partitions / np.power(2, self.level_from_top - 1)
        if p_cnt == 1:  # leaf node cannot split further
            return
        
        left_id = PartitionNode.allocate_id()
        left = PartitionNode(left_id, iso_l, mu_l, self.key_range_start, self.capacity // 2)
        left.father = self
        left.level_from_top = self.level_from_top + 1
        left.micro_partition_features = torch.zeros(self.max_micro_partitions, 4)
        for i in range(int(p_cnt // 2)):
            left.micro_partition_features[int(self.p_start + i)] = self.micro_partition_features[int(self.p_start + i)]
        
        right_id = PartitionNode.allocate_id()
        right = PartitionNode(right_id, iso_r, mu_r, self.key_range_start + self.capacity // 2, self.capacity // 2)
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


    def merge(self, iso, mu):
        if not self.can_merge:
            print("Error: cannot merge non-mergeable partition")
            return
        
        self.is_leaf = True
        self.can_merge = False  # now a leaf — cannot merge again
        if self.father is not None:
            self.father.can_merge = True
        
        self.left = None
        self.right = None
        self.isolation_level = iso
        self.mu = mu
