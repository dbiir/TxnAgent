import math

from torch.nn import functional as F

from agent.partition import PartitionNode


class HeuristicSelector:
    def __init__(self, lambda_param: float = 0.7):
        self.lambda_param = lambda_param  # Decay rate for exp(-λ·Δt)
        self.execution_history = {}  # Track when partitions were last selected
        self.current_time_step = 0
        
    def calculate_partition_score(self, p: PartitionNode):
        """Calculate selection score S = w·I + (1-w)·D
        where w = exp(-λ·Δt), Δt = steps since last selected.
        """
        dt = self.current_time_step - self.execution_history.get(p.p_id, 0)
        w = math.exp(-self.lambda_param * dt)

        intensity_score = p.workload_intensity
        
        # Workload diversity component (cosine similarity)
        if p.previous_embedding is not None:
            diversity = 1 - F.cosine_similarity(p.current_embedding, p.previous_embedding) / 2
        else:
            diversity = 0.0

        score = w * intensity_score + (1 - w) * diversity
        return score
    
    def topK(self, partitions: list[PartitionNode], K: int) -> list[PartitionNode]:
        scores: dict[int, float] = {}
        for p in partitions:
            scores[p.p_id] = self.calculate_partition_score(p)
        
        # Select top-K partitions based on scores
        selected_partitions = sorted(scores, key=scores.get, reverse=True)[:K]
        
        # Record selection time for chosen partitions
        for pid in selected_partitions:
            self.execution_history[pid] = self.current_time_step
        self.current_time_step += 1

        result: list[PartitionNode] = []
        for p in partitions:
            if p.p_id in selected_partitions:
                result.append(p)

        return result