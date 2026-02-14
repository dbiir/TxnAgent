# TxnAgent Architecture

## Overview

The RL agent dynamically adjusts database partition configurations (isolation levels, μ, splits/merges) to maximize throughput while minimizing abort rates. It uses **MAML (Model-Agnostic Meta-Learning)** for fast online adaptation to changing workloads.

### System Architecture

```
StatisticsWorker (Java)          adapter.py            TxnAgent (Python)
┌─────────────────────┐   TCP    ┌──────────┐          ┌──────────────┐
│ Collect per-partition│──:7654──►│ Parse    │─────────►│ service()    │
│ stats every 5s      │         │ "online" │          │ RL decision  │
│                     │◄────────│ Forward  │◄─────────│ id#iso#mu    │
│ applyActions()      │ response│          │          │              │
└─────────────────────┘         └──────────┘          └──────────────┘
```

### Startup Order
1. `adapter.py` — starts TCP server on `:7654`
2. `TxnSailsServer` — Java server, `StatisticsWorker` connects to adapter
3. `TriStar` — benchmark client, drives workload

On shutdown, Java sends `"close"` → Python exports metrics, saves checkpoint, exits.

---

## Data Structures

### PartitionNode (Python: `partition.py`)

```python
class PartitionNode:
    p_id: int                    # partition ID
    isolation_level: int         # 0=RC, 1=SI, 2=SER
    mu: int                      # timestamp interval parameter
    p_start: int                 # micro-partition range start
    capacity: int                # number of micro-partitions (8)
    is_leaf: bool                # can split / change_iso / adjust_interval
    can_merge: bool              # can merge children
    left: PartitionNode          # left child (after split)
    right: PartitionNode         # right child (after split)
    micro_partition_features: Tensor  # shape [8, 4]: per-micro features
    workload_intensity: float
    current_embedding: Tensor
    previous_embedding: Tensor
```

**Micro-partition features** (4 per micro-partition):
- `read_ratio` = read_count / (read_count + write_count)
- `abort_ratio`
- `workload_intensity`
- `isolation_level`

### PartitionGraph (Python: `graph.py`)

```python
class PartitionGraph:
    nodes: Dict[int, PartitionNode]        # partition_id → node
    edges: Dict[Tuple[int,int], int]        # (i,j) → distributed txn count
    tput: float                             # global throughput
    abort_ratio: float                      # global abort rate
```

**Configuration**: 8 macro partitions × 8 micro-partitions = 64 micro-partitions, each holding up to 1024 keys.

---

## Part I: State Representation

Two modes controlled by `USE_GRAPH_EMBEDDING` flag (default: **False**):

| Mode | Features (32-dim) | Description |
|------|-------------------|-------------|
| **Direct** (`False`) | `partition.get_node_features()` | Flattened micro-partition features: 8 × 4 = 32 |
| **GNN** (`True`) | `GraphEmbeddingModel` output | GNN encodes graph structure into embeddings |

**State vector** (36-dim):
```
state = [features(32), isolation_level, mu, is_leaf, can_merge]
```

---

## Part II: Partition Selection

Only **leaf** and **can_merge** nodes are candidates.

**Heuristic scoring** (`heuristic.py`):
$$S_{p_i} = \lambda \cdot I + (1 - \lambda) \cdot D$$

- $I$ = workload intensity
- $D$ = 1 − cosine_similarity(current_embedding, previous_embedding) / 2
- $\lambda = \frac{2}{e^t + e^{-t}}$ (decays over time: prioritizes intensity early, diversity later)

Most valuable partition is selected for adjustment each iteration.

---

## Part III: RL Action Selection

### Actor Network (`rl_model.py → MultiHeadParameterizedActor`)

```
state(36) → shared_trunk(128→64) → action_logits(4)
                                  → split_head(64→6)  → iso_l, mu_l, iso_r, mu_r
                                  → merge_head(64→4)  → iso, mu
                                  → iso_head(64→3)    → iso (Categorical)
                                  → interval_head(64→2) → mu (Normal)
```

**Action masking**: invalid actions are masked to $-\infty$ before softmax.

### Actions

| # | Action | Constraint | Parameters |
|---|--------|-----------|------------|
| 0 | `split` | `is_leaf=True` | iso_l, μ_l, iso_r, μ_r |
| 1 | `merge` | `can_merge=True` | iso, μ |
| 2 | `change_iso` | `is_leaf=True` | iso (0=RC, 1=SI, 2=SER) |
| 3 | `adjust_interval` | `is_leaf=True` | μ (integer) |

### Critic Network

```
state(36) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(1) → value
```

---

## Part IV: Reward Function

$$R = \alpha \cdot R_p - (1 - \alpha) \cdot P_c$$

**Performance gain:**
$$R_p = \eta_p \cdot \frac{P_t - P_0}{P_0} + (1 - \eta_p) \cdot \frac{P_t - P_{t-1}}{P_{t-1}}$$

**Correctness penalty**
$$P_c = \eta_c \cdot \frac{C_t - C_0}{C_0} + (1 - \eta_c) \cdot \frac{C_t - C_{t-1}}{C_{t-1}}$$

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| α | 0.7 | Weight performance vs correctness |
| η_p | 0.5 | Long-term vs short-term in R_p |
| η_c | 0.5 | Long-term vs short-term in P_c |

---

## Part V: MAML Training

### MetaPPO (`rl_model.py`)

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| inner_lr | 1e-3 | Inner loop (adaptation) learning rate |
| meta_lr | 3e-4 | Outer loop (meta) learning rate |
| clip_eps | 0.2 | PPO clipping range |
| γ | 0.99 | Discount factor |
| λ | 0.95 | GAE lambda |
| entropy_coef | 0.01 | Entropy bonus |

### Online Adaptation (`agent.py → service()`)

```
for each service() call:
    1. Parse sample file → graph + features
    2. Compute reward from throughput/abort delta
    3. Store experience in buffer
    4. If buffer ≥ 4:  inner_update(buffer) → adapted_params
    5. Select action using adapted_params
    6. If buffer ≥ 16: meta_update(support, query) → clear buffer
    7. Return response: "id#iso#mu;id#iso#mu;..."
```

### Offline Meta-Training (`offline_train_rl.py`)

```
for each epoch:
    for each workload directory (= MAML task):
        1. Collect rollout from sample files
        2. Split into support / query sets
    meta_update(task_batch)  # adapt on support, evaluate on query
    update_critic(query_states, returns)
    Save best checkpoint by meta-loss
```

**Checkpoint**: `checkpoints/best_meta_ppo.pt`

---

## Communication Protocol

### Java → Python (`StatisticsWorker`)

Sample file header: `nodeCount#edgeCount#throughput#abortRate`  
Node line: `id#readRatio#abortRatio#workloadIntensity#isolationLevel`  
Edge line: `src#dst#count`  

Socket message: `"online,<filepath>"`  
Shutdown message: `"close"`

### Python → Java

Response: `"id#iso#mu;id#iso#mu;..."` — one entry per micro-partition.

Java `applyActions()` parses each entry and calls:
- `PartitionManager.setIsolation(id, level)`
- `PartitionManager.setMu(id, mu)`

---

## Observability

### TensorBoard (`runs/online_rl/`, `runs/offline_meta_train/`)

| Panel | Metrics |
|-------|---------|
| `step/*` | reward, throughput, abort_cost, tput_vs_baseline |
| `update/*` | meta_loss, critic_loss, advantage_mean/std, avg_reward |
| `epoch/*` | meta_loss, critic_loss, best_loss (offline only) |

### Metrics Export

On close, `export_metrics()` writes `logs/metrics/metrics_<YYYYMMDD_HHMMSS>.json` containing all per-step and per-update metrics.

```bash
tensorboard --logdir runs/
```