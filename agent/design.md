# Architecture of TxnAgent
each partition includes 8 micro partitions and each micro partition holds up to 1024 keys.

## Basic Data Structure for Partition
A global data structure for managing partition information (act as a tree):
```Java
class Partition {
    private int isolationLevel;      // 0:SER, 1:SI, 2:RC
    private int size;               // number of micro partitions, initially 8
    private int keyRangeStart;
    private int mu;                 // parameter for timestamp interval adjustment
    private float workloadIntensity; // transaction access frequency
    private boolean isLeaf;
    private boolean canMerge;
    private List<Float> previousEmbedding; // vector embedding representation
    private Partition left;         // left child partition
    private Partition right;        // right child partition
    private Partition parent;      // parent partition
}
```

## Part I: Workload Embedding
Graph (V, A, E)
1. V: set of vertices, each node represents a partition
   - [write/read ratio, contention, anomaly count, isolation level] * 8
2. A: set of edge attributes
   - edge weight: distributed transaction count across two partitions
3. E: set of edges, each edge represents transactions between two partitions
   - undirected edge from partition i to partition j if there are transactions between them

Goal: learn a function f: (V, A, E) -> Performance

Output: each partition's embedding vector

## Part II: Partition Selection
### Principles
- initial phase: prioritize partition with high workload intensity $I$
- stable phase: prioritize partitions with high workload diversity $D$, $D$ is defined as cosine similarity between current embedding and previous embedding

### Calculate score of each partition
- compute score $S$ for each partition:
  $$ S_{p_i} = \lambda I + (1-\lambda) / 2 \cdot D$$

$\delta t$: time elapsed since execution, $e^{-\delta t}$

### Select top-k partitions with highest scores for adjustment
- sort partitions by score $S$ in descending order
- select top-k partitions as candidates for adjustment

## Part III: Adjustment
For each selected partition, use a PPO model to choose one action and apply it

### Action
1. (idx, split, $\emptyset$): [only leaf partitions (isLeaf=true)]
    - split the partition into two child partitions with equal key ranges, inheriting isolation level and $\mu$ from parent
    - add child partitions to adjustment candidate set
    - mark its _isLeaf_ as false; _canMerge_ as true, mark its parent partition's _canMerge_ as false;

2. (idx, merge, iso, $\mu$): [only the father partition of two leaf partitions (canMerge=true)]
    - merge two child partitions into one parent partition with key range covering both children
    - remove child partitions from adjustment candidate set
    - set isolation level and $\mu$ of parent partition as specified
    - mark its _isLeaf_ as true; _canMerge_ as false; mark its parent partition's _canMerge_ as true;

3. (idx, iso, SER/SI/RC): [only leaf partitions (isLeaf=true)]
    - change the isolation level of the partition to the specified level

4. (idx, interval, param): [only leaf partitions (isLeaf=true)]
    - adjust the timestamp interval parameter $\mu$ of the partition to the specified value

### Reward Function
After adjustment of all selected partitions, clear the adjustment candidate set. Get the reward based on the overall performance.
We consider the following metrics:
    - Performance improvement $R_p$: $$R_p = \eta_p \frac{P_t - P_0}{P_0} + (1 - \eta_p) \frac{P_t - P_{t-1}}{P_{t-1}}$$
    - Penalty violations of correctness $P_c$: $$P_c = \eta_c \frac{C_t - C_0}{C_0} + (1 - \eta_c) \frac{C_t - C_{t-1}}{C_{t-1}}, C = \sum_{i=1}^{N}c_i$$

Reward function: 
    $$R = \alpha \cdot R_p - (1 - \alpha) \cdot P_c$$

### Procedure
Meta-learning model 
Input: partition embedding, is_leaf, can_merge
Output: an action from the above 4 types, use a mask to filter invalid actions

**Offline training**
Generate many training data from different kinds of workloads

**Online execution**
Fine-tune update the network parameters based on real-time feedback