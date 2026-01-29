from agent.heuristic import HeuristicSelector
from agent.partition import PartitionGraph, PartitionNode
import pandas as pd

class TxnAgent:
    def __init__(self):
        self.max_micro_partitions = 8
        self.partition_info: list[PartitionNode] = []
        self.graph: PartitionGraph = PartitionGraph()
        self.heuristic_selector = HeuristicSelector()

    def init_partition_info(self, count: int, capacity: int):
        for i in range(count):
            self.partition_info.append(PartitionNode(i, 0, 2, i * capacity, capacity=capacity))

    def service(self, f_partition: str, f_edges: str) -> str:
        # load workload characteristics
        self.load_partition_features(f_partition)
        
        # constuct the graph
        self.load_edge_features(f_edges)
        
        # embedding
        
        # select partitions
        
        # reinforce learning for each partition
        
        pass
    
    """
    Recevice the csv file path of the partition info, format as follows:
        idx, write/read ratio, w_contention, r_contention, anomaly count, workload intensity
    """
    def load_partition_features(self, csv_path: str):
        wrkld = self.read_partition_csv(csv_path)
        if wrkld is None:
            return
        for i, row in wrkld.iterrows():
            features = [row['write/read ratio'], row['w_contention'], row['r_contention'], row['anomaly count'], row['workload intensity']]
            self.partition_info[i // self.max_micro_partitions].update_micro_partition_features(i % self.max_micro_partitions, features)
            
        # add partition node
        for partition in self.partition_info:
            self.recursive_add_node(partition)

    def recursive_add_node(self, p: PartitionNode):
        if p.can_merge:
            self.graph.add_partition(p)
            self.graph.add_partition(p.left)
            self.graph.add_partition(p.right)
            self.graph.add_edge(p.p_id, p.left.p_id)
            self.graph.add_edge(p.p_id, p.right.p_id)
            return
        else:
            self.recursive_add_node(p.left)
            self.recursive_add_node(p.right)
    
    """
    p_id1, p_id2, transaction count
    """
    def load_edge_features(self, csv_path: str):
        wrkld = self.read_partition_csv(csv_path)
        if wrkld is None:
            return
        for _, row in wrkld.iterrows():
            p1 = self.partition_info[row['p_id1'] // self.max_micro_partitions].find_leaf_partition(row['p_id1'] % self.max_micro_partitions)
            p2 = self.partition_info[row['p_id2'] // self.max_micro_partitions].find_leaf_partition(row['p_id2'] % self.max_micro_partitions)
            self.graph.add_edge(p1.p_id, p2.p_id, row['transaction count'])
    
    def read_partition_csv(csv_path) -> pd.DataFrame | None:
        try:
            # read csv file
            data = pd.read_csv(csv_path)
            
            print(f"number of partition: {len(data)}")
            print(f"number of column: {list(data.columns)}")
            return data

        except FileNotFoundError:
            print(f"File does not exist - {csv_path}")
            return None
        except Exception as e:
            print(f"Read csv failed: {e}")
            return None