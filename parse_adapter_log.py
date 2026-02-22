#!/usr/bin/env python3
"""
Parse adapter log and extract RL training metrics.

Usage:
    python parse_adapter_log.py <logfile>
    python parse_adapter_log.py <logfile> --plot       # also produce plots
    python parse_adapter_log.py <logfile> --csv out.csv # export to CSV
"""

import re
import sys
import argparse
import json
from collections import Counter


def parse_log(filepath: str) -> dict:
    """Parse adapter log and return structured metrics."""

    iterations = []       # per-iteration records
    maml_updates = []     # MAML update records
    warmup_iters = []     # warm-up iterations (no reward)

    # Regex patterns
    re_reward = re.compile(
        r'\[Iter (\d+)\] reward=([-\d.]+)\s+tput=([\d.]+)\s+abort_cost=([\d.]+)'
    )
    re_baseline = re.compile(
        r'\[Iter (\d+)\] baseline set\s+tput=([\d.]+)\s+abort_cost=([\d.]+)'
    )
    re_warmup = re.compile(
        r'\[Iter (\d+)\] warm-up \(tput=([\d.]+)'
    )
    re_action = re.compile(
        r"\[Iter (\d+)\] partition=(\d+)\s+action=(\w+)\s+params=(\{[^}]*\})\s+adapted=(yes|no)"
    )
    re_maml = re.compile(
        r'\[Iter (\d+)\] MAML update: meta_loss=([-\d.]+)\s+critic=([-\d.]+)\s+buffer_size=(\d+)'
    )
    re_timing = re.compile(
        r'\[Iter (\d+)\] TIMING\s+total=([\d.]+)ms\s+load=([\d.]+)ms\s+embed=([\d.]+)ms\s+'
        r'reward=([\d.]+)ms\s+heuristic=([\d.]+)ms\s+maml=([\d.]+)ms\s+action=([\d.]+)ms'
    )

    # Index by iteration for merging
    iter_data = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Warm-up
            m = re_warmup.match(line)
            if m:
                it = int(m.group(1))
                warmup_iters.append(it)
                d = iter_data.setdefault(it, {'iter': it})
                d['phase'] = 'warmup'
                d['tput'] = float(m.group(2))
                continue

            # Baseline
            m = re_baseline.match(line)
            if m:
                it = int(m.group(1))
                d = iter_data.setdefault(it, {'iter': it})
                d['phase'] = 'baseline'
                d['tput'] = float(m.group(2))
                d['abort_cost'] = float(m.group(3))
                continue

            # Reward line
            m = re_reward.match(line)
            if m:
                it = int(m.group(1))
                d = iter_data.setdefault(it, {'iter': it})
                d['phase'] = 'active'
                d['reward'] = float(m.group(2))
                d['tput'] = float(m.group(3))
                d['abort_cost'] = float(m.group(4))
                continue

            # Action line
            m = re_action.match(line)
            if m:
                it = int(m.group(1))
                d = iter_data.setdefault(it, {'iter': it})
                d['partition'] = int(m.group(2))
                d['action'] = m.group(3)
                d['params'] = m.group(4)
                d['adapted'] = m.group(5) == 'yes'
                continue

            # MAML update
            m = re_maml.match(line)
            if m:
                maml_updates.append({
                    'iter': int(m.group(1)),
                    'meta_loss': float(m.group(2)),
                    'critic_loss': float(m.group(3)),
                    'buffer_size': int(m.group(4)),
                })
                continue

            # Timing
            m = re_timing.match(line)
            if m:
                it = int(m.group(1))
                d = iter_data.setdefault(it, {'iter': it})
                d['timing'] = {
                    'total_ms': float(m.group(2)),
                    'load_ms': float(m.group(3)),
                    'embed_ms': float(m.group(4)),
                    'reward_ms': float(m.group(5)),
                    'heuristic_ms': float(m.group(6)),
                    'maml_ms': float(m.group(7)),
                    'action_ms': float(m.group(8)),
                }
                continue

    # Sort by iteration
    iterations = [iter_data[k] for k in sorted(iter_data.keys())]

    return {
        'iterations': iterations,
        'maml_updates': maml_updates,
        'warmup_count': len(warmup_iters),
    }


def print_summary(data: dict):
    """Print a human-readable summary of the parsed metrics."""
    iters = data['iterations']
    maml = data['maml_updates']

    active = [d for d in iters if d.get('phase') == 'active']
    rewards = [d['reward'] for d in active]
    tputs = [d['tput'] for d in active]
    aborts = [d['abort_cost'] for d in active]
    actions = [d.get('action', '?') for d in iters if 'action' in d]

    print(f"{'='*60}")
    print(f" Adapter Log Summary")
    print(f"{'='*60}")
    print(f"  Total iterations:    {len(iters)}")
    print(f"  Warm-up iterations:  {data['warmup_count']}")
    print(f"  Active iterations:   {len(active)}")
    print(f"  MAML updates:        {len(maml)}")
    print()

    if rewards:
        print(f"  Reward   — min: {min(rewards):.4f}  max: {max(rewards):.4f}  "
              f"avg: {sum(rewards)/len(rewards):.4f}")
    if tputs:
        print(f"  Tput     — min: {min(tputs):.1f}  max: {max(tputs):.1f}  "
              f"avg: {sum(tputs)/len(tputs):.1f}")
    if aborts:
        print(f"  Abort    — min: {min(aborts):.4f}  max: {max(aborts):.4f}  "
              f"avg: {sum(aborts)/len(aborts):.4f}")

    if actions:
        print(f"\n  Action distribution:")
        for action, cnt in Counter(actions).most_common():
            print(f"    {action:20s}  {cnt:4d}  ({cnt/len(actions)*100:.1f}%)")

    adapted_cnt = sum(1 for d in iters if d.get('adapted'))
    total_action = sum(1 for d in iters if 'adapted' in d)
    if total_action:
        print(f"\n  MAML adapted ratio:  {adapted_cnt}/{total_action} "
              f"({adapted_cnt/total_action*100:.1f}%)")

    if maml:
        losses = [m['meta_loss'] for m in maml]
        critics = [m['critic_loss'] for m in maml]
        print(f"\n  Meta-loss — min: {min(losses):.4f}  max: {max(losses):.4f}  "
              f"avg: {sum(losses)/len(losses):.4f}")
        print(f"  Critic    — min: {min(critics):.4f}  max: {max(critics):.4f}  "
              f"avg: {sum(critics)/len(critics):.4f}")

    print(f"{'='*60}")

    # Print series
    print(f"\n{'Iter':>5s}  {'Tput':>10s}  {'Abort':>8s}  {'Reward':>8s}  "
          f"{'Action':>18s}  {'Adapted':>7s}")
    print('-' * 65)
    for d in iters:
        phase = d.get('phase', '')
        tput = f"{d['tput']:.1f}" if 'tput' in d else ''
        abort = f"{d.get('abort_cost', 0):.4f}" if 'abort_cost' in d else ''
        reward = f"{d['reward']:.4f}" if 'reward' in d else phase
        action = d.get('action', '')
        adapted = 'yes' if d.get('adapted') else ('no' if 'adapted' in d else '')
        print(f"{d['iter']:5d}  {tput:>10s}  {abort:>8s}  {reward:>8s}  "
              f"{action:>18s}  {adapted:>7s}")


def export_csv(data: dict, filepath: str):
    """Export parsed data to CSV."""
    import csv
    iters = data['iterations']
    maml_by_iter = {m['iter']: m for m in data['maml_updates']}

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'iter', 'phase', 'tput', 'abort_cost', 'reward',
            'action', 'partition', 'adapted',
            'meta_loss', 'critic_loss'
        ])
        for d in iters:
            m = maml_by_iter.get(d['iter'], {})
            writer.writerow([
                d['iter'],
                d.get('phase', ''),
                d.get('tput', ''),
                d.get('abort_cost', ''),
                d.get('reward', ''),
                d.get('action', ''),
                d.get('partition', ''),
                d.get('adapted', ''),
                m.get('meta_loss', ''),
                m.get('critic_loss', ''),
            ])
    print(f"CSV exported to: {filepath}")


def plot_metrics(data: dict):
    """Plot throughput, reward, and abort cost series."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    iters = data['iterations']
    active = [d for d in iters if d.get('phase') == 'active']

    if not active:
        print("No active iterations to plot")
        return

    xs = [d['iter'] for d in active]
    rewards = [d['reward'] for d in active]
    tputs = [d['tput'] for d in active]
    aborts = [d['abort_cost'] for d in active]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Throughput
    axes[0].plot(xs, tputs, 'b-o', markersize=2, linewidth=1)
    axes[0].set_ylabel('Throughput (req/s)')
    axes[0].set_title('RL Training Metrics')
    axes[0].grid(True, alpha=0.3)

    # Reward
    axes[1].plot(xs, rewards, 'g-o', markersize=2, linewidth=1)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Reward')
    axes[1].grid(True, alpha=0.3)

    # Abort cost
    axes[2].plot(xs, aborts, 'r-o', markersize=2, linewidth=1)
    axes[2].set_ylabel('Abort Cost')
    axes[2].set_xlabel('Iteration')
    axes[2].grid(True, alpha=0.3)

    # Mark MAML updates
    for m in data['maml_updates']:
        for ax in axes:
            ax.axvline(x=m['iter'], color='purple', linestyle=':', alpha=0.3)

    plt.tight_layout()
    out_path = 'adapter_metrics.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to: {out_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse adapter log for RL metrics')
    parser.add_argument('logfile', help='Path to adapter log file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--csv', type=str, default=None, help='Export to CSV file')
    parser.add_argument('--json', type=str, default=None, help='Export to JSON file')
    args = parser.parse_args()

    data = parse_log(args.logfile)
    print_summary(data)

    if args.csv:
        export_csv(data, args.csv)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"JSON exported to: {args.json}")

    if args.plot:
        plot_metrics(data)
