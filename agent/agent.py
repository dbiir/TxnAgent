import random
import torch
from agent.graph import PartitionGraph
from agent.graph_embedding import GraphEmbeddingModel
from agent.heuristic import HeuristicSelector
from agent.partition import PartitionNode
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
import pandas as pd
import os

class TxnAgent:
    def __init__(self):
        self.partition_cnt = 8
        self.max_micro_partitions = 8
        self.marco_partition_capacity = 8192
        self.partition_info: list[PartitionNode] = []   # used for online self-adaptive adjustment
        self.graphs: list[PartitionGraph]
        self.heuristic_selector = HeuristicSelector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_encoder = None  # to be initialized later
        self.heuristic_selector = None
        
        self.init_partition_info(self.partition_cnt, self.marco_partition_capacity)
        
    def init_partition_info(self, count: int, capacity: int):
        for i in range(count):
            self.partition_info.append(PartitionNode(i, 0, 2, i * capacity, capacity=capacity))


    def service(self, filename: str, workload: str = None):
        if self.embedding_model is None:      
            self.graph_encoder = self.load_model("graph_encoder.pt", self.device)
        if self.heuristic_selector is None:
            self.heuristic_selector = HeuristicSelector()

        # load workload characteristics
        graph = PartitionGraph()
        self.load_graph_from_file(graph, filename)
        node_emb = self.get_partition_embeddings(graph)
        self.update_partition_embeddings(graph, node_emb)
        partitions: list[PartitionNode] = [n for n in graph.nodes.values()]
        # select the adjust partition
        partition_candidates = self.heuristic_selector.topK(partitions, K=1)  # select most valuable partition to adjust
        
        # TODO: to the rl agent for action decision
        
    
    def update_partition_embeddings(self, graph: PartitionGraph, node_emb: torch.Tensor):
        node_ids = sorted(graph.nodes.keys())

        for idx, nid in enumerate(node_ids):
            graph.nodes[nid].update_embedding(node_emb[idx].detach().cpu())


    def offline_service(self, filepath: str):
        # traverse the files in the filepath and load the data
        dataset = []
        for f in os.listdir(filepath):
            if f.startswith("sample"):
                graph = PartitionGraph()
                full_path = os.path.join(filepath, f)
                self.load_graph_from_file(graph, full_path)
                self.graphs.append(graph)
                data = graph.to_pyg_data()
                dataset.append(data)
        
        if len(dataset) == 0:
            raise RuntimeError("No training samples found")
        
        train_dataset = dataset[: int(0.9 * len(dataset))]
        val_dataset = dataset[int(0.9 * len(dataset)) :]
        
        if self.embedding_model is None:
            self.embedding_model = GraphEmbeddingModel(
                node_in_dim=32,
                edge_in_dim=1,
                hidden_dim=256,
                output_dim=128,
                num_layers=3
            )
            self.embedding_model.to(self.device)

        self.offline_train(
            model=self.embedding_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=self.device,
            epochs=100,
            batch_size=64,
            lr=1e-3,
        )


    def offline_train(self, model, train_dataset, val_dataset, device, epochs=100, batch_size=32, lr=1e-3):
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        best_val = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, device)
            val_loss = self.eval_epoch(model, val_loader, device)

            print(
                f"[Epoch {epoch:03d}] "
                f"train={train_loss:.4f} "
                f"val={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                self.save_model(model, "graph_encoder.pt")


    def train_epoch(self, model, loader, optimizer, device):
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            _, pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            target = batch.y.view(-1)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    @torch.no_grad()
    def eval_epoch(self, model, loader, device):
        model.eval()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)

            _, pred = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch
            )

            target = batch.y.view(-1)
            loss = F.mse_loss(pred, target)

            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)


    def save_model(self, model, path: str):
        checkpoint = {
            "model_state": model.state_dict(),
            "node_in_dim": model.node_in_dim,
            "edge_in_dim": model.edge_in_dim,
            "hidden_dim": model.hidden_dim,
            "output_dim": model.output_dim,
            "num_layers": len(model.local_layers),
        }
        torch.save(checkpoint, path)


    @torch.no_grad()
    def get_partition_embeddings(self, graph: PartitionGraph, device: torch.device):
        data = graph.to_pyg_data()
        data = data.to(device)

        node_emb, _ = self.embedding_model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch
        )

        return node_emb  # [num_partitions, embedding_dim]

    def load_model(path: str, device: torch.device):
        checkpoint = torch.load(path, map_location=device)

        model = GraphEmbeddingModel(
            node_in_dim=checkpoint["node_in_dim"],
            edge_in_dim=checkpoint["edge_in_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            output_dim=checkpoint["output_dim"],
            num_layers=checkpoint["num_layers"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()  # VERY IMPORTANT

        return model


    def load_graph_from_file(self, graph: PartitionGraph, filename: str, partition_info: list[PartitionNode] = []):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError("Empty file")
        
        if len(partition_info) == 0:
            for i in range(self.partition_cnt):
                node = PartitionNode(i, 0, 2, i * 8192)
                if random.random() < 0.5:
                    node.split()
                    if random.random() < 0.5:
                        node.left.split()
                    if random.random() < 0.5:
                        node.right.split()
                partition_info.append(node)
        
        # ---- parse head ----
        try:
            head_parts = lines[0].split("#")
            node_count=int(head_parts[0])
            edge_count=int(head_parts[1])
            tput=float(head_parts[2])
            abort=float(head_parts[3])
            graph.set_features(tput, abort)
        except Exception as e:
            raise ValueError(f"Invalid head line: {lines[0]}") from e

        # ---- parse nodes ----
        idx = 1
        for i in range(node_count):
            if idx >= len(lines):
                raise ValueError("Unexpected EOF while reading nodes")

            parts = lines[idx].split("#")
            if len(parts) != 6:
                raise ValueError(f"Invalid node line: {lines[idx]}")

            id=int(parts[0])
            if id != i:
                raise ValueError(f"Node ID mismatch: expected {i}, got {id}")
            
            rcnt=int(parts[1])
            wcnt=int(parts[2])
            abort_ratio=float(parts[3])
            workload_intensive=float(parts[4])
            isolation_level=int(parts[5]) / 2.0
            features = [rcnt / (rcnt + wcnt), abort_ratio, workload_intensive, isolation_level]
            partition_info[i // self.max_micro_partitions].update_micro_partition_features(i % self.max_micro_partitions, features)
            idx += 1
            
        # add partition node
        for partition in partition_info:
            self.recursive_add_node(graph, partition)

        # ---- parse edges ----
        for _ in range(edge_count):
            if idx >= len(lines):
                raise ValueError("Unexpected EOF while reading edges")

            parts = lines[idx].split("#")
            if len(parts) != 3:
                raise ValueError(f"Invalid edge line: {lines[idx]}")

            src=int(parts[0])
            dst=int(parts[1])
            count=int(parts[2])
            p1 = partition_info[src // self.max_micro_partitions].find_leaf_partition(src % self.max_micro_partitions)
            p2 = partition_info[dst // self.max_micro_partitions].find_leaf_partition(dst % self.max_micro_partitions)
            graph.add_edge(p1.p_id, p2.p_id, count)
            idx += 1


    def recursive_add_node(self, graph: PartitionGraph, p: PartitionNode):
        if p.can_merge:
            graph.add_partition(p)
            graph.add_partition(p.left)
            graph.add_partition(p.right)
            graph.add_edge(p.p_id, p.left.p_id)
            graph.add_edge(p.p_id, p.right.p_id)
        else:
            self.recursive_add_node(p.left)
            self.recursive_add_node(p.right)


    # currently not used
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