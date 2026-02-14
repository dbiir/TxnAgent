"""
Offline MAML Meta-Training for Partition RL Agent.

Trains the MetaPPO agent across multiple workload types using MAML,
so the resulting model can quickly adapt to new workloads online.

Usage:
    python -m agent.offline_train_rl --data_dir data/workloads --epochs 200

Directory layout expected:
    data/workloads/
        oltp/           <- one workload = one MAML "task"
            sample_001.txt
            sample_002.txt
            ...
        olap/
            sample_001.txt
            ...
        mixed/
            ...
"""

import os
import sys
import argparse
import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from agent.agent import TxnAgent
from agent.rl_model import MetaPPO, compute_gae
from agent.graph import PartitionGraph
from agent.partition import PartitionNode


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_workloads(data_dir: str) -> dict:
    """Discover workload directories. Each subdirectory = one MAML task.
    Returns {workload_name: [file_paths]}."""
    workloads = {}
    for entry in sorted(os.listdir(data_dir)):
        subdir = os.path.join(data_dir, entry)
        if not os.path.isdir(subdir):
            continue
        files = sorted([
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if f.startswith("sample") and f.endswith(".txt")
        ])
        if files:
            workloads[entry] = files
    return workloads


def collect_rollout(agent: TxnAgent, meta_ppo: MetaPPO,
                    file_paths: list, device: torch.device):
    """Run the agent over a sequence of workload snapshots and collect
    the 7-tuple that inner_update/meta_update expect.

    Returns
    -------
    task_data : tuple (states, action_types, action_params,
                       action_masks, rewards, dones, old_log_probs)
    all_states, all_returns : for critic updates
    """
    states_list = []
    action_types_list = []
    action_masks_list = []
    log_probs_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    param_accum = {'iso_l': [], 'mu_l': [], 'iso_r': [], 'mu_r': [],
                   'iso': [], 'mu': []}

    prev_tput = None
    prev_abort = None
    baseline_tput = None
    baseline_abort = None

    for step_idx, filepath in enumerate(file_paths):
        # Load graph + embeddings
        graph = PartitionGraph()
        partition_info = []
        for i in range(agent.partition_cnt):
            partition_info.append(
                PartitionNode(i, 0, 2, i * agent.marco_partition_capacity,
                              capacity=agent.marco_partition_capacity)
            )
        agent.load_graph_from_file(graph, filepath, partition_info)

        current_tput = graph.tput
        current_abort = graph.abort_ratio

        if agent.graph_encoder is None:
            agent.graph_encoder = agent.load_model("graph_encoder.pt", device)

        node_emb = agent.get_partition_embeddings(graph, device)

        node_ids = sorted(graph.nodes.keys())
        embeddings = {
            nid: node_emb[idx].detach().cpu()
            for idx, nid in enumerate(node_ids)
        }

        # Compute reward
        if baseline_tput is None:
            baseline_tput = current_tput
            baseline_abort = current_abort

        if prev_tput is not None and step_idx > 0:
            eta_p, eta_c, alpha = agent.ETA_P, agent.ETA_C, agent.REWARD_ALPHA
            p0 = max(baseline_tput, 1e-8)
            p_prev = max(prev_tput, 1e-8)
            r_p = eta_p * (current_tput - baseline_tput) / p0 \
                + (1 - eta_p) * (current_tput - prev_tput) / p_prev

            c0 = max(baseline_abort, 1e-8)
            c_prev = max(prev_abort, 1e-8)
            p_c = eta_c * (current_abort - baseline_abort) / c0 \
                + (1 - eta_c) * (current_abort - prev_abort) / c_prev

            reward = alpha * r_p - (1 - alpha) * p_c
        else:
            reward = 0.0

        prev_tput = current_tput
        prev_abort = current_abort

        # Select partition and get RL action
        partitions = list(graph.nodes.values())
        if agent.heuristic_selector is None:
            from agent.heuristic import HeuristicSelector
            agent.heuristic_selector = HeuristicSelector()
        candidates = agent.heuristic_selector.topK(partitions, K=1)

        for partition in candidates:
            if partition.p_id not in embeddings:
                continue

            state = agent.prepare_state(partition, embeddings[partition.p_id])

            action_mask = agent.generate_action_mask(partition)

            with torch.no_grad():
                dist, param_dists = meta_ppo.actor.get_distributions(
                    state.unsqueeze(0), action_mask.unsqueeze(0)
                )
                action = dist.sample()
                action_idx = action.item()
                log_prob = dist.log_prob(action)

                value = meta_ppo.critic(state.unsqueeze(0)).squeeze().item()

            action_name = agent.ACTION_TYPES[action_idx]
            parameters = agent._sample_action_parameters(action_name, param_dists)
            agent.execute_action(partition, action_name, parameters)

            # Accumulate
            states_list.append(state)
            action_types_list.append(action_idx)
            action_masks_list.append(action_mask)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            dones_list.append(0.0)
            values_list.append(value)

            for key in param_accum:
                param_accum[key].append(parameters.get(key, 0.0))

    if len(states_list) == 0:
        return None, None, None

    # Build tensors
    states = torch.stack(states_list)
    action_types = torch.tensor(action_types_list, dtype=torch.long)
    action_masks = torch.stack(action_masks_list)
    old_log_probs = torch.stack(log_probs_list).squeeze()
    action_params = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in param_accum.items()
    }

    # task_data: the 7-tuple expected by inner_update / meta_update
    task_data = (states, action_types, action_params, action_masks,
                 rewards_list, dones_list, old_log_probs)

    # For critic updates
    advantages, returns = compute_gae(rewards_list, values_list, dones_list)

    return task_data, states, returns


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover workloads
    workloads = discover_workloads(args.data_dir)
    if len(workloads) == 0:
        print(f"No workload directories found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(workloads)} workloads: {list(workloads.keys())}")
    for name, files in workloads.items():
        print(f"  {name}: {len(files)} samples")

    # Initialize agent + MetaPPO
    agent = TxnAgent()
    meta_ppo = agent.rl_agent  # type: MetaPPO

    # Checkpoint directory
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss = float('inf')

    # TensorBoard
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', 'offline_meta_train')
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.epochs):
        task_batch = []
        all_states = []
        all_returns = []

        for wk_name, wk_files in workloads.items():
            # Shuffle and split into support/query
            shuffled = wk_files.copy()
            random.shuffle(shuffled)
            mid = max(1, len(shuffled) // 2)
            support_files = shuffled[:mid]
            query_files = shuffled[mid:]

            # Collect rollouts
            support_data, s_states, s_returns = collect_rollout(
                agent, meta_ppo, support_files, device
            )
            query_data, q_states, q_returns = collect_rollout(
                agent, meta_ppo, query_files, device
            )

            if support_data is None or query_data is None:
                continue

            task_batch.append((support_data, query_data))
            all_states.append(q_states)
            all_returns.append(q_returns)

        if len(task_batch) == 0:
            print(f"[Epoch {epoch+1}] No valid tasks, skipping")
            continue

        # MAML meta-update
        meta_loss = meta_ppo.meta_update(task_batch)

        # Critic update
        cat_states = torch.cat(all_states)
        cat_returns = torch.cat(all_returns)
        critic_loss = meta_ppo.update_critic(cat_states, cat_returns)

        print(f"[Epoch {epoch+1:3d}/{args.epochs}]  "
              f"meta_loss={meta_loss:.4f}  critic_loss={critic_loss:.4f}")

        # TensorBoard logging
        writer.add_scalar('epoch/meta_loss', meta_loss, epoch + 1)
        writer.add_scalar('epoch/critic_loss', critic_loss, epoch + 1)
        writer.add_scalar('epoch/num_tasks', len(task_batch), epoch + 1)

        # Save best checkpoint
        if meta_loss < best_loss:
            best_loss = meta_loss
            save_path = os.path.join(ckpt_dir, 'best_meta_ppo.pt')
            meta_ppo.save(save_path)
            print(f"  -> saved best model (loss={best_loss:.4f})")
            writer.add_scalar('epoch/best_loss', best_loss, epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(ckpt_dir, f'meta_ppo_epoch_{epoch+1}.pt')
            meta_ppo.save(save_path)

    # Final save
    final_path = os.path.join(ckpt_dir, 'meta_ppo_final.pt')
    meta_ppo.save(final_path)
    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML Meta-Training for Partition RL')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing workload subdirectories')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of meta-training epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
