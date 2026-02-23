import random
import torch
import yaml
from agent.graph import PartitionGraph
from agent.graph_embedding import GraphEmbeddingModel
from agent.heuristic import HeuristicSelector
from agent.partition import PartitionNode
from agent.rl_model import MetaPPO, MultiHeadParameterizedActor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

class TxnAgent:
    # State dimension: 32 (graph embedding) + 4 (iso, mu, is_leaf, can_merge)
    STATE_DIM = 36
    ACTION_TYPES = ['split', 'merge', 'change_iso', 'adjust_interval']

    REWARD_ALPHA = 0.7   # α: weight performance vs correctness in final reward
    ETA_P = 0.5          # η_p: weight long-term vs short-term in R_p
    ETA_C = 0.5          # η_c: weight long-term vs short-term in P_c

    USE_GRAPH_EMBEDDING = False  # True = GNN embedding, False = raw partition features
    MIN_BASELINE_TPUT = 100      # minimum throughput (req/s) before establishing baseline

    def __init__(self, workload: str = 'ycsb'):
        self.workload = workload
        self.max_micro_partitions = 8

        # Read partition layout from YAML
        partition_yaml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'config', 'partition', workload, 'partition.yaml'
        )
        self.partition_cnt, self.marco_partition_capacity = \
            self._load_partition_config(partition_yaml)

        self.partition_info: list[PartitionNode] = []
        self.graphs: list[PartitionGraph]
        self.heuristic_selector = HeuristicSelector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph_encoder = None  # to be initialized later
        self.heuristic_selector = None

        # RL agent for partition adjustment decisions
        self.rl_agent = MetaPPO(state_dim=self.STATE_DIM)

        # Try loading saved checkpoints (online > offline)
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'models'
        )
        online_ckpt = os.path.join(models_dir, 'final_online.pt')
        meta_ckpt = os.path.join(models_dir, 'best_meta_ppo.pt')

        if os.path.exists(online_ckpt):
            self.rl_agent.load(online_ckpt)
            print(f"[TxnAgent] Resumed from online checkpoint: {online_ckpt}", flush=True)
        elif os.path.exists(meta_ckpt):
            self.rl_agent.load(meta_ckpt)
            print(f"[TxnAgent] Loaded meta-trained checkpoint: {meta_ckpt}", flush=True)
        else:
            print("[TxnAgent] No checkpoint found, starting fresh", flush=True)

        # Cross-iteration RL state tracking
        self.prev_tput = None
        self.prev_abort_cost = None
        self.baseline_tput = None
        self.baseline_abort_cost = None
        self.prev_state = None
        self.prev_action_idx = None
        self.prev_action_params = None
        self.prev_log_prob = None
        self.prev_action_mask = None
        self.iteration = 0

        # MAML online adaptation
        self.adapted_params = None
        self.adaptation_steps = 4

        # Experience buffer for online updates
        self.update_interval = 8
        self.experience_buffer = {
            'states': [],
            'action_types': [],
            'action_params': {'iso_l': [], 'mu_l': [], 'iso_r': [], 'mu_r': [],
                              'iso': [], 'mu': []},
            'action_masks': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }

        # Transition buffer for offline training: list of (s, a, r, s') dicts
        self.transition_buffer = []

        # Config output directory
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'partition')
        os.makedirs(self.config_dir, exist_ok=True)

        # TensorBoard logging
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'online_rl')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.update_count = 0  # tracks number of MAML updates

        # In-memory metrics for file export on close
        self.metrics_history = {
            'step_reward': [],
            'step_throughput': [],
            'step_abort_cost': [],
            'step_tput_vs_baseline': [],
            'update_meta_loss': [],
            'update_critic_loss': [],
            'update_advantage_mean': [],
            'update_avg_reward': [],
        }

        self.init_partition_info()

    def _load_partition_config(self, yaml_path: str):
        """Read partition YAML and compute partition_cnt and marco_partition_capacity.

        partition_cnt = sum(relation.partitionCount for all relations) / 8
        marco_partition_capacity = {macro_idx: partitionSize * 8}
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        workload_data = data.get('workload', data)
        relations = workload_data.get('relations', [])

        total_micro_partitions = 0
        marco_capacity = {}  # macro_partition_idx -> capacity (partitionSize * 8)
        macro_idx = 0

        for rel in relations:
            partition_size = rel['partitionSize']
            partition_count = rel['partitionCount']
            total_micro_partitions += partition_count
            # Each macro partition groups 8 micro partitions
            num_macro = partition_count // 8
            capacity = partition_size * 8
            for _ in range(num_macro):
                marco_capacity[macro_idx] = capacity
                macro_idx += 1

        partition_cnt = total_micro_partitions // 8
        print(f"[TxnAgent] Loaded partition config: {partition_cnt} macro partitions, "
              f"capacities={marco_capacity}", flush=True)
        return partition_cnt, marco_capacity

    def init_partition_info(self):
        key_offset = 0
        for i in range(self.partition_cnt):
            capacity = self.marco_partition_capacity[i]
            self.partition_info.append(
                PartitionNode(i, 0, 2, key_offset, capacity=capacity)
            )
            key_offset += capacity


    def service(self, filename: str, workload: str = None) -> str:
        import time as _time
        t_start = _time.perf_counter()

        if self.heuristic_selector is None:
            self.heuristic_selector = HeuristicSelector()

        self.iteration += 1

        # 1. Load workload characteristics
        t0 = _time.perf_counter()
        graph = PartitionGraph()
        self.load_graph_from_file(graph, filename, self.partition_info)
        current_tput = graph.tput
        current_abort = graph.abort_ratio
        t_load = _time.perf_counter() - t0

        # Build feature lookup: partition_id -> feature tensor
        t0 = _time.perf_counter()
        if self.USE_GRAPH_EMBEDDING:
            # GNN mode: compute graph embeddings
            if self.graph_encoder is None:
                self.graph_encoder = self.load_model("graph_encoder.pt", self.device)
            node_emb = self.get_partition_embeddings(graph, self.device)
            self.update_partition_embeddings(graph, node_emb)
            node_ids = sorted(graph.nodes.keys())
            embeddings = {
                nid: node_emb[idx].detach().cpu()
                for idx, nid in enumerate(node_ids)
            }
        else:
            # Direct mode: use raw partition features (8 micro × 4 feats = 32-dim)
            embeddings = {
                nid: graph.nodes[nid].get_node_features()
                for nid in graph.nodes
            }
        t_embed = _time.perf_counter() - t0

        # 2. Reward calculation
        t0 = _time.perf_counter()
        if self.baseline_tput is None:
            if current_tput >= self.MIN_BASELINE_TPUT:
                # System has warmed up — establish baselines P_0, C_0
                self.baseline_tput = current_tput
                self.baseline_abort_cost = current_abort
                print(f"[Iter {self.iteration}] baseline set  "
                      f"tput={current_tput:.1f}  abort_cost={current_abort:.4f}", flush=True)
            else:
                print(f"[Iter {self.iteration}] warm-up (tput={current_tput:.1f} "
                      f"< {self.MIN_BASELINE_TPUT}), skipping", flush=True)
        elif self.prev_state is not None:
            reward = self.calculate_reward(current_tput, current_abort)
            print(f"[Iter {self.iteration}] reward={reward:.4f}  "
                  f"tput={current_tput:.1f}  abort_cost={current_abort:.4f}", flush=True)

            # Log per-step metrics
            self.writer.add_scalar('step/reward', reward, self.iteration)
            self.writer.add_scalar('step/throughput', current_tput, self.iteration)
            self.writer.add_scalar('step/abort_cost', current_abort, self.iteration)
            self.metrics_history['step_reward'].append((self.iteration, reward))
            self.metrics_history['step_throughput'].append((self.iteration, current_tput))
            self.metrics_history['step_abort_cost'].append((self.iteration, current_abort))
            if self.baseline_tput:
                tput_ratio = (current_tput - self.baseline_tput) / max(self.baseline_tput, 1e-8)
                self.writer.add_scalar('step/tput_vs_baseline', tput_ratio, self.iteration)
                self.metrics_history['step_tput_vs_baseline'].append((self.iteration, tput_ratio))

            # Store experience from previous step
            self.store_experience(reward, done=False)

            # Record (s, a, r, s') transition for offline training
            current_state = None
            for partition in [p for p in graph.nodes.values() if p.is_leaf or p.can_merge]:
                if partition.p_id in embeddings:
                    current_state = self.prepare_state(partition, embeddings[partition.p_id])
                    break
            self.transition_buffer.append({
                'state': self.prev_state.detach().clone(),
                'action_type': self.prev_action_idx,
                'action_params': dict(self.prev_action_params),
                'action_mask': self.prev_action_mask.detach().clone(),
                'log_prob': self.prev_log_prob.detach().clone(),
                'reward': reward,
                'next_state': current_state.detach().clone() if current_state is not None else self.prev_state.detach().clone(),
                'done': False,
            })

            # Trigger PPO update when buffer is full
            if len(self.experience_buffer['rewards']) >= self.update_interval:
                self.update_rl_agent()
        else:
            print(f"[Iter {self.iteration}] "
                  f"tput={current_tput:.1f}  abort_cost={current_abort:.4f}", flush=True)
        t_reward = _time.perf_counter() - t0

        # 3. Select the most valuable partition to adjust
        t0 = _time.perf_counter()
        partitions = [p for p in graph.nodes.values() if p.is_leaf or p.can_merge]

        # Print per-partition isolation + overall metrics
        iso_names = {0: 'RC', 1: 'SI', 2: 'SER'}
        leaf_parts = [p for p in graph.nodes.values() if p.is_leaf]
        iso_summary = '  '.join(f"p{p.p_id}={iso_names.get(p.isolation_level, p.isolation_level)}"
                                for p in sorted(leaf_parts, key=lambda x: x.p_id))
        print(f"[Iter {self.iteration}] tput={graph.tput:.1f}  abort={graph.abort_ratio:.4f}  "
              f"partitions=[{iso_summary}]", flush=True)

        partition_candidates = self.heuristic_selector.topK(partitions, K=1)
        t_heuristic = _time.perf_counter() - t0

        # 4. MAML adaptation: use buffered experience as support set
        t0 = _time.perf_counter()
        buf_size = len(self.experience_buffer['rewards'])
        if buf_size >= self.adaptation_steps:
            support_data = self._build_task_data()
            self.adapted_params = self.rl_agent.inner_update(support_data)
        else:
            self.adapted_params = None
        t_maml = _time.perf_counter() - t0

        # 5. RL action decision + execution (using adapted params)
        t0 = _time.perf_counter()
        for partition in partition_candidates:
            if partition.p_id not in embeddings:
                continue

            state = self.prepare_state(partition, embeddings[partition.p_id])
            action_mask = self.generate_action_mask(partition)

            action_type_dist, param_dists = self.rl_agent.actor.get_distributions(
                state.unsqueeze(0), action_mask.unsqueeze(0),
                params=self.adapted_params  # MAML-adapted params (or None)
            )
            action_type = action_type_dist.sample()
            action_idx = action_type.item()
            action_name = self.ACTION_TYPES[action_idx]

            parameters = self._sample_action_parameters(action_name, param_dists)

            # Compute full log-prob (action type + parameter) for PPO ratio
            type_lp = action_type_dist.log_prob(action_type)
            param_lp = self._compute_action_param_log_prob(
                action_idx, param_dists, parameters
            )
            log_prob = type_lp + param_lp

            self.execute_action(partition, action_name, parameters)

            print(f"[Iter {self.iteration}] partition={partition.p_id}  "
                  f"action={action_name}  params={parameters}  "
                  f"adapted={'yes' if self.adapted_params else 'no'}", flush=True)

            # Save state for reward calculation on next call
            self.prev_state = state
            self.prev_action_idx = action_idx
            self.prev_action_params = parameters
            self.prev_log_prob = log_prob.detach()
            self.prev_action_mask = action_mask
        t_action = _time.perf_counter() - t0

        # 6. Update metrics for next iteration
        self.prev_tput = current_tput
        self.prev_abort_cost = current_abort

        t_total = _time.perf_counter() - t_start
        print(f"[Iter {self.iteration}] TIMING  total={t_total*1000:.1f}ms  "
              f"load={t_load*1000:.1f}ms  embed={t_embed*1000:.1f}ms  "
              f"reward={t_reward*1000:.1f}ms  heuristic={t_heuristic*1000:.1f}ms  "
              f"maml={t_maml*1000:.1f}ms  action={t_action*1000:.1f}ms", flush=True)

        # 7. Build response string for Java applyActions()
        return self.format_response() + '\n'


    # ----------------------------------------------------------------
    # Experience buffer & online RL update
    # ----------------------------------------------------------------

    def store_experience(self, reward: float, done: bool = False):
        """Push the previous step's (state, action, reward, ...) into the buffer."""
        buf = self.experience_buffer
        buf['states'].append(self.prev_state)
        buf['action_types'].append(self.prev_action_idx)
        buf['action_masks'].append(self.prev_action_mask)
        buf['log_probs'].append(self.prev_log_prob)
        buf['rewards'].append(reward)
        buf['dones'].append(float(done))

        # Critic value for GAE
        with torch.no_grad():
            value = self.rl_agent.critic(self.prev_state.unsqueeze(0)).squeeze().item()
        buf['values'].append(value)

        # Buffer action params (fill 0 for unused params)
        params = self.prev_action_params
        for key in buf['action_params']:
            if key in params:
                buf['action_params'][key].append(params[key])
            else:
                buf['action_params'][key].append(0.0)

    def _build_task_data(self):
        """Convert the experience buffer into the 7-tuple format that
        inner_update / meta_update expect.

        Returns
        -------
        (states, action_types, action_params, action_masks,
         rewards, dones, old_log_probs)
        """
        buf = self.experience_buffer
        states = torch.stack(buf['states'])
        action_types = torch.tensor(buf['action_types'], dtype=torch.long)
        action_masks = torch.stack(buf['action_masks'])
        old_log_probs = torch.stack(buf['log_probs']).squeeze()
        action_params = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in buf['action_params'].items()
        }
        return (states, action_types, action_params, action_masks,
                buf['rewards'], buf['dones'], old_log_probs)

    def _clear_buffer(self):
        """Reset the experience buffer."""
        self.experience_buffer = {
            'states': [],
            'action_types': [],
            'action_params': {'iso_l': [], 'mu_l': [], 'iso_r': [], 'mu_r': [],
                              'iso': [], 'mu': []},
            'action_masks': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
        }

    def update_rl_agent(self):
        """Run MAML-style meta-update using buffered experiences.
        Split buffer into support (first half) and query (second half),
        inner-adapt on support, evaluate on query, meta-update."""
        buf = self.experience_buffer
        n = len(buf['rewards'])
        if n < 2:
            return

        # Build full task data and split into support/query
        full_data = self._build_task_data()
        mid = n // 2

        def _slice(data, start, end):
            """Slice the 7-tuple."""
            states, atypes, aparams, amasks, rewards, dones, log_probs = data
            return (
                states[start:end],
                atypes[start:end],
                {k: v[start:end] for k, v in aparams.items()},
                amasks[start:end],
                rewards[start:end],
                dones[start:end],
                log_probs[start:end],
            )

        support_data = _slice(full_data, 0, mid)
        query_data = _slice(full_data, mid, n)

        # MAML meta-update: inner-adapt on support, evaluate on query
        task_batch = [(support_data, query_data)]
        meta_loss = self.rl_agent.meta_update(task_batch)

        # Critic update on query states
        q_states = full_data[0][mid:n]
        from agent.rl_model import compute_gae
        advantages, returns = compute_gae(
            buf['rewards'][mid:], buf['values'][mid:], buf['dones'][mid:]
        )
        critic_loss = self.rl_agent.update_critic(q_states, returns)

        print(f"[Iter {self.iteration}] MAML update: "
              f"meta_loss={meta_loss:.4f}  critic={critic_loss:.4f}  "
              f"buffer_size={n}", flush=True)

        # Log update metrics to TensorBoard
        self.update_count += 1
        self.writer.add_scalar('update/meta_loss', meta_loss, self.update_count)
        self.writer.add_scalar('update/critic_loss', critic_loss, self.update_count)
        self.writer.add_scalar('update/advantage_mean', advantages.mean().item(), self.update_count)
        self.writer.add_scalar('update/advantage_std', advantages.std().item(), self.update_count)
        self.writer.add_scalar('update/buffer_size', n, self.update_count)
        avg_reward = sum(buf['rewards']) / n
        self.writer.add_scalar('update/avg_reward', avg_reward, self.update_count)

        self.metrics_history['update_meta_loss'].append((self.update_count, meta_loss))
        self.metrics_history['update_critic_loss'].append((self.update_count, critic_loss))
        self.metrics_history['update_advantage_mean'].append((self.update_count, advantages.mean().item()))
        self.metrics_history['update_avg_reward'].append((self.update_count, avg_reward))

        # Clear buffer and reset adaptation
        self._clear_buffer()
        self.adapted_params = None

    def export_metrics(self) -> str:
        """Export training metrics to a timestamped JSON file. Returns the file path."""
        import json
        from datetime import datetime

        metrics_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(metrics_dir, f'metrics_{timestamp}.json')

        export = {
            'total_iterations': self.iteration,
            'total_updates': self.update_count,
            'baseline_tput': self.baseline_tput,
            'baseline_abort_cost': self.baseline_abort_cost,
            'metrics': {
                key: [{'step': s, 'value': float(v)} for s, v in values]
                for key, values in self.metrics_history.items()
            },
        }

        with open(filepath, 'w') as f:
            json.dump(export, f, indent=2)

        print(f"Metrics exported to: {filepath}", flush=True)

        # Also save transitions for offline training
        self.save_transitions()

        return filepath

    def save_transitions(self):
        """Save recorded (s, a, r, s') transitions to disk for offline MAML training."""
        if len(self.transition_buffer) == 0:
            print("No transitions to save.", flush=True)
            return

        from datetime import datetime
        trans_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'metas', 'transitions'
        )
        os.makedirs(trans_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(trans_dir, f'transitions_{timestamp}.pt')
        torch.save(self.transition_buffer, filepath)
        print(f"Saved {len(self.transition_buffer)} transitions to: {filepath}", flush=True)

    @staticmethod
    def _safe_ratio(current: float, baseline: float) -> float:
        """Compute (current - baseline) / baseline safely.

        When baseline is near zero (< 1.0), return the raw difference
        instead of dividing by a tiny number, which would blow up the
        reward to astronomical values.
        """
        diff = current - baseline
        if abs(baseline) < 1.0:
            return diff  # raw difference, bounded by the metric range
        return diff / baseline

    def calculate_reward(self, current_tput: float, current_abort_cost: float) -> float:
        """Paper reward: R = α · R_p - (1-α) · P_c

        R_p = η_p · (P_t - P_0)/P_0 + (1-η_p) · (P_t - P_{t-1})/P_{t-1}
        P_c = η_c · (C_t - C_0)/C_0 + (1-η_c) · (C_t - C_{t-1})/C_{t-1}

        When baseline or previous values are near zero, raw differences
        are used instead of ratios to keep rewards in a stable range.
        """
        alpha = self.REWARD_ALPHA
        eta_p = self.ETA_P
        eta_c = self.ETA_C

        # Performance gain R_p (long-term + short-term)
        r_p = eta_p * self._safe_ratio(current_tput, self.baseline_tput) \
            + (1 - eta_p) * self._safe_ratio(current_tput, self.prev_tput)

        # Correctness violation P_c (long-term + short-term)
        p_c = eta_c * self._safe_ratio(current_abort_cost, self.baseline_abort_cost) \
            + (1 - eta_c) * self._safe_ratio(current_abort_cost, self.prev_abort_cost)

        return alpha * r_p - (1 - alpha) * p_c

    def prepare_state(self, partition: PartitionNode, embedding: torch.Tensor) -> torch.Tensor:
        """Build state vector: [embedding(32), iso_level, mu, is_leaf, can_merge]"""
        return torch.cat([
            embedding,
            torch.tensor([
                float(partition.isolation_level),
                float(partition.mu),
                float(partition.is_leaf),
                float(partition.can_merge)
            ])
        ])

    def generate_action_mask(self, partition: PartitionNode) -> torch.Tensor:
        """Generate action mask based on partition constraints.
        [split, merge, change_iso, adjust_interval]"""
        mask = torch.zeros(4)
        if partition.is_leaf:
            mask[0] = 1  # split allowed for leaf nodes
            mask[2] = 1  # change_iso allowed for leaf nodes
            mask[3] = 1  # adjust_interval allowed for leaf nodes
        if partition.can_merge:
            mask[1] = 1  # merge allowed when children exist
        return mask

    @torch.no_grad()
    def _sample_action_parameters(self, action_name: str, param_dists: dict) -> dict:
        """Sample concrete parameter values from the RL actor's distributions."""
        if action_name == 'split':
            d = param_dists['split']
            return {
                'iso_l': d['iso_l'].sample().item(),
                'mu_l': max(1, int(d['mu_l'].sample().item())),
                'iso_r': d['iso_r'].sample().item(),
                'mu_r': max(1, int(d['mu_r'].sample().item())),
            }
        elif action_name == 'merge':
            d = param_dists['merge']
            return {
                'iso': d['iso'].sample().item(),
                'mu': max(1, int(d['mu'].sample().item())),
            }
        elif action_name == 'change_iso':
            d = param_dists['change_iso']
            return {'iso': d['iso'].sample().item()}
        elif action_name == 'adjust_interval':
            d = param_dists['adjust_interval']
            return {'mu': max(1, int(d['mu'].sample().item()))}
        return {}

    @torch.no_grad()
    def _compute_action_param_log_prob(self, action_idx: int,
                                        param_dists: dict,
                                        params: dict) -> torch.Tensor:
        """Compute parameter log-prob for a single sampled action (B=1).
        This mirrors what _compute_param_log_prob does in batch, but for
        a single sample at action-selection time."""
        name = self.ACTION_TYPES[action_idx]
        dists = param_dists[name]
        lp = torch.tensor(0.0)

        if name == 'split':
            lp = (dists['iso_l'].log_prob(torch.tensor(params['iso_l']))[0]
                  + dists['mu_l'].log_prob(torch.tensor([[params['mu_l']]], dtype=torch.float32))[0].sum()
                  + dists['iso_r'].log_prob(torch.tensor(params['iso_r']))[0]
                  + dists['mu_r'].log_prob(torch.tensor([[params['mu_r']]], dtype=torch.float32))[0].sum())
        elif name == 'merge':
            lp = (dists['iso'].log_prob(torch.tensor(params['iso']))[0]
                  + dists['mu'].log_prob(torch.tensor([[params['mu']]], dtype=torch.float32))[0].sum())
        elif name == 'change_iso':
            lp = dists['iso'].log_prob(torch.tensor(params['iso']))[0]
        elif name == 'adjust_interval':
            lp = dists['mu'].log_prob(torch.tensor([[params['mu']]], dtype=torch.float32))[0].sum()

        return lp

    def execute_action(self, partition: PartitionNode, action_name: str, params: dict):
        """Execute the RL-decided action on a partition."""
        if action_name == 'split' and partition.is_leaf:
            iso_l = int(params.get('iso_l', 0))
            mu_l = int(params.get('mu_l', 2))
            iso_r = int(params.get('iso_r', 0))
            mu_r = int(params.get('mu_r', 2))
            partition.split(iso_l, mu_l, iso_r, mu_r)

        elif action_name == 'merge' and partition.can_merge:
            iso = int(params.get('iso', partition.isolation_level))
            mu = int(params.get('mu', partition.mu))
            partition.isolation_level = iso
            partition.mu = mu
            partition.merge(iso, mu)

        elif action_name == 'change_iso' and partition.is_leaf:
            iso = int(params.get('iso', partition.isolation_level))
            partition.isolation_level = iso

        elif action_name == 'adjust_interval' and partition.is_leaf:
            mu = int(params.get('mu', partition.mu))
            partition.mu = mu

    # ----------------------------------------------------------------
    # Response formatting for Java applyActions()
    # ----------------------------------------------------------------

    def format_response(self) -> str:
        """Build response string for Java applyActions().
        Format: id#iso#mu;id#iso#mu;...
        One entry per micro-partition."""
        parts = []
        for macro in self.partition_info:
            for offset in range(self.max_micro_partitions):
                idx = macro.p_id * self.max_micro_partitions + offset
                leaf = macro.find_leaf_partition(offset)
                iso = leaf.isolation_level if leaf is not None else macro.isolation_level
                mu = leaf.mu if leaf is not None else macro.mu
                parts.append(f'{idx}#{int(iso)}#{int(mu)}')
        return ';'.join(parts)

    # ----------------------------------------------------------------
    # Embedding helpers
    # ----------------------------------------------------------------

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
                output_dim=32,
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
        with open(filename.strip(), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError("Empty file")
        
        if len(partition_info) == 0:
            for i in range(self.partition_cnt):
                capacity = self.marco_partition_capacity[i]
                node = PartitionNode(i, 0, 2, i * capacity, capacity=capacity)
                if random.random() < 0.5:
                    node.split(0, 2, 0, 2)
                    if random.random() < 0.5:
                        node.left.split(0, 2, 0, 2)
                    if random.random() < 0.5:
                        node.right.split(0, 2, 0, 2)
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
        # format: id#readRatio#abortRatio#workloadIntensity#isolationLevel
        idx = 1
        for i in range(node_count):
            if idx >= len(lines):
                raise ValueError("Unexpected EOF while reading nodes")

            parts = lines[idx].split("#")
            if len(parts) != 5:
                raise ValueError(f"Invalid node line (expected 5 fields): {lines[idx]}")

            id = int(parts[0])
            if id != i:
                raise ValueError(f"Node ID mismatch: expected {i}, got {id}")

            read_ratio = float(parts[1])
            abort_ratio = float(parts[2])
            workload_intensity = float(parts[3])
            isolation_level = int(parts[4]) / 2.0
            features = [read_ratio, abort_ratio, workload_intensity, isolation_level]
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
        if p.is_leaf:
            graph.add_partition(p)
            return
        if p.can_merge:
            graph.add_partition(p)
            graph.add_partition(p.left)
            graph.add_partition(p.right)
            graph.add_edge(p.p_id, p.left.p_id)
            graph.add_edge(p.p_id, p.right.p_id)
        else:
            self.recursive_add_node(graph, p.left)
            self.recursive_add_node(graph, p.right)
