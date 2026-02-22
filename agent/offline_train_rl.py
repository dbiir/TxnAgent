"""
Offline MAML Meta-Training for Partition RL Agent.

Trains the MetaPPO agent across multiple workload types using MAML,
so the resulting model can quickly adapt to new workloads online.

The training data consists of (s, a, r, s') transition files (.pt)
recorded during online operation by TxnAgent.save_transitions().

Usage:
    python -m agent.offline_train_rl --data_dir metas/transitions --epochs 200

Directory layout expected:
    metas/transitions/
        transitions_20260221_120000.pt   <- one .pt file per online run
        transitions_20260221_130000.pt
        ...

Each .pt file contains a list of dicts with keys:
    state, action_type, action_params, action_mask, log_prob, reward, next_state, done
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def discover_transition_files(data_dir: str) -> list:
    """Find all transition .pt files in the data directory.
    Returns list of file paths."""
    files = []
    for f in sorted(os.listdir(data_dir)):
        if f.startswith("transitions_") and f.endswith(".pt"):
            files.append(os.path.join(data_dir, f))
    return files


def load_transitions(filepath: str) -> list:
    """Load a single transition file.
    Returns list of (s, a, r, s') dicts."""
    transitions = torch.load(filepath, weights_only=False)
    return transitions


def build_task_data_from_transitions(transitions: list, device: torch.device,
                                     meta_ppo: MetaPPO):
    """Convert a list of recorded (s, a, r, s') transitions into the
    7-tuple format that inner_update / meta_update expect.

    Returns
    -------
    task_data : tuple (states, action_types, action_params,
                       action_masks, rewards, dones, old_log_probs)
    all_states : Tensor of states for critic updates
    all_returns : Tensor of discounted returns
    """
    if len(transitions) == 0:
        return None, None, None

    states_list = []
    action_types_list = []
    action_masks_list = []
    log_probs_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    param_accum = {'iso_l': [], 'mu_l': [], 'iso_r': [], 'mu_r': [],
                   'iso': [], 'mu': []}

    for t in transitions:
        state = t['state'].to(device)
        action_type = t['action_type']
        action_params = t['action_params']
        action_mask = t['action_mask'].to(device)
        reward = t['reward']
        done = t.get('done', False)

        # Re-evaluate log_prob under current policy for importance weighting
        with torch.no_grad():
            dist, _ = meta_ppo.actor.get_distributions(
                state.unsqueeze(0), action_mask.unsqueeze(0)
            )
            action_tensor = torch.tensor([action_type], device=device)
            log_prob = dist.log_prob(action_tensor)
            value = meta_ppo.critic(state.unsqueeze(0)).squeeze().item()

        states_list.append(state.cpu())
        action_types_list.append(action_type)
        action_masks_list.append(action_mask.cpu())
        log_probs_list.append(log_prob.cpu())
        rewards_list.append(reward)
        dones_list.append(float(done))
        values_list.append(value)

        for key in param_accum:
            param_accum[key].append(action_params.get(key, 0.0))

    # Build tensors
    states = torch.stack(states_list)
    action_types = torch.tensor(action_types_list, dtype=torch.long)
    action_masks = torch.stack(action_masks_list)
    old_log_probs = torch.stack(log_probs_list).squeeze()
    action_params_tensors = {
        k: torch.tensor(v, dtype=torch.float32) for k, v in param_accum.items()
    }

    # task_data: the 7-tuple expected by inner_update / meta_update
    task_data = (states, action_types, action_params_tensors, action_masks,
                 rewards_list, dones_list, old_log_probs)

    # For critic updates
    advantages, returns = compute_gae(rewards_list, values_list, dones_list)

    return task_data, states, returns


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Discover transition files
    trans_files = discover_transition_files(args.data_dir)
    if len(trans_files) == 0:
        print(f"No transition files found in {args.data_dir}")
        sys.exit(1)

    # Load all transitions and group them as MAML tasks
    # Each .pt file is one online run = one MAML task
    all_tasks = []
    for fpath in trans_files:
        transitions = load_transitions(fpath)
        if len(transitions) >= args.min_transitions:
            all_tasks.append((os.path.basename(fpath), transitions))
            print(f"  Loaded {len(transitions):4d} transitions from {os.path.basename(fpath)}")
        else:
            print(f"  Skipped {os.path.basename(fpath)} ({len(transitions)} < {args.min_transitions})")

    if len(all_tasks) == 0:
        print("No valid tasks found (need at least --min_transitions per file)")
        sys.exit(1)

    print(f"\nLoaded {len(all_tasks)} tasks with "
          f"{sum(len(t) for _, t in all_tasks)} total transitions")

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

        for task_name, transitions in all_tasks:
            # Shuffle and split into support/query
            shuffled = transitions.copy()
            random.shuffle(shuffled)
            mid = max(1, len(shuffled) // 2)
            support_trans = shuffled[:mid]
            query_trans = shuffled[mid:]

            # Build task data from recorded transitions
            support_data, s_states, s_returns = build_task_data_from_transitions(
                support_trans, device, meta_ppo
            )
            query_data, q_states, q_returns = build_task_data_from_transitions(
                query_trans, device, meta_ppo
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
                        help='Directory containing transitions_*.pt files')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of meta-training epochs')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--min_transitions', type=int, default=8,
                        help='Minimum transitions per file to be used as a task')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
