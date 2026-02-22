import math
import time
from torch.nn import functional as F

from agent.partition import PartitionNode


class HeuristicSelector:
    def __init__(self):
        self.execution_time = time.time_ns()
        self.execution_history = {}  # Track when partitions were last selected
        self.current_time_step = 0
        
    def get_lambda(self):
        time_escaped = (time.time_ns() - self.execution_time) / 1e9
        # sech(t) = 2 / (e^t + e^-t); clamp to avoid overflow (sech(700) ≈ 0)
        time_escaped = min(time_escaped, 700)
        return 2 / (math.exp(time_escaped) + math.exp(-time_escaped))
        
    def calculate_partition_score(self, p: PartitionNode):
        """Calculate selection score S = λI + (1-λ)/2 * D"""
        # Workload intensity component
        lambda_param = self.get_lambda()
        intensity_score = p.workload_intensity
        
        # Workload diversity component (cosine similarity)
        if p.previous_embedding is not None:
            diversity = 1 - F.cosine_similarity(p.current_embedding, p.previous_embedding) / 2
        else:
            diversity = 0.0

        # Combined score
        score = lambda_param * intensity_score + (1 - lambda_param) * diversity
        
        return score
    
    def topK(self, partitions: list[PartitionNode], K: int) -> list[PartitionNode]:
        scores: dict[int, float] = {}
        for p in partitions:
            scores[p.p_id] = self.calculate_partition_score(p)
        
        # Select top-K partitions based on scores
        selected_partitions = sorted(scores, key=scores.get, reverse=True)[:K]
        
        result: list[PartitionNode] = []
        for p in partitions:
            if p.p_id in selected_partitions:
                result.append(p)

        return result