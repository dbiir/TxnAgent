import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import deque
import random

from agent.partition import PartitionNode

class PPOBuffer:
    """Experience replay buffer for PPO with parameterized actions"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.action_masks = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, log_prob, value, action_mask):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.action_masks.append(action_mask)
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer"""
        if len(self.states) < batch_size:
            return None
            
        indices = random.sample(range(len(self.states)), batch_size)
        
        batch = {
            'states': [self.states[i] for i in indices],
            'actions': [self.actions[i] for i in indices],
            'rewards': [self.rewards[i] for i in indices],
            'next_states': [self.next_states[i] for i in indices],
            'log_probs': [self.log_probs[i] for i in indices],
            'values': [self.values[i] for i in indices],
            'action_masks': [self.action_masks[i] for i in indices],
        }
        
        return batch
    
    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.log_probs.clear()
        self.values.clear()
        self.action_masks.clear()
    
    def __len__(self):
        return len(self.states)

class MultiHeadParameterizedActor(nn.Module):
    """Multi-head actor network that generates both action types and parameters"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(MultiHeadParameterizedActor, self).__init__()
        
        # Shared feature extractor
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action type head (discrete: 4 action types)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 action types
            nn.Softmax(dim=-1)
        )
        
        # Parameter heads for each action type
        self.parameter_heads = nn.ModuleDict({
            # split action: no parameters needed
            'split': nn.Identity(),  # No parameters for split
            
            # merge action: isolation_level (discrete) + mu (continuous)
            'merge': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 4)  # iso_level (3) + mu_bin (1)
            ),
            
            # change_iso action: target isolation level (discrete)
            'change_iso': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # 3 isolation levels
            ),
            
            # adjust_interval action: new mu value (continuous)
            'adjust_interval': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),  # single continuous value
                nn.Sigmoid()  # Normalize to [0, 1] range
            )
        })
    
    def forward(self, state: torch.Tensor) -> Dict:
        """Forward pass generating both action type and parameters"""
        shared_features = self.shared_network(state)
        
        # Get action type probabilities
        action_probs = self.action_type_head(shared_features)
        
        # Get parameters for each action type
        parameters = {}
        for action_name, head in self.parameter_heads.items():
            raw_params = head(shared_features)
            parameters[action_name] = self._process_parameters(action_name, raw_params)
        
        return {
            'action_probs': action_probs,
            'parameters': parameters
        }
    
    def _process_parameters(self, action_name: str, raw_params: torch.Tensor) -> Dict:
        """Process raw parameters into meaningful values"""
        if action_name == 'split':
            return {}  # No parameters needed
        
        elif action_name == 'merge':
            # iso_level: discrete (0, 1, 2), mu: continuous
            iso_logits = raw_params[:, :3] if len(raw_params.shape) > 1 else raw_params[:3]
            mu_value = raw_params[:, 3:4] if len(raw_params.shape) > 1 else raw_params[3:4]
            mu_value = mu_value * 10.0  # Scale to reasonable range
            
            return {
                'isolation_level_probs': F.softmax(iso_logits, dim=-1),
                'mu_value': mu_value
            }
        
        elif action_name == 'change_iso':
            # Target isolation level probabilities
            return {'target_iso_probs': F.softmax(raw_params, dim=-1)}
        
        elif action_name == 'adjust_interval':
            # Scaled mu value (0.1 to 10.0 range)
            scaled_mu = raw_params * 9.9 + 0.1  # Scale to [0.1, 10.0]
            return {'new_mu': scaled_mu}
        
        return {}

class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ParameterizedPPOAgent:
    """Complete PPO agent with multi-head parameterized actor"""
    
    def __init__(self, state_dim: int, action_dim: int = 4, 
                 hidden_dim: int = 128, lr: float = 3e-4, 
                 gamma: float = 0.99, clip_epsilon: float = 0.2,
                 ppo_epochs: int = 4, batch_size: int = 32):
        
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Networks
        self.actor = MultiHeadParameterizedActor(state_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = PPOBuffer()
        
        # Action space
        self.action_types = ['split', 'merge', 'change_iso', 'adjust_interval']
        self.isolation_levels = ['SER', 'SI', 'RC']  # 0, 1, 2
    
    def select_parameterized_action(self, state: torch.Tensor, action_mask: torch.Tensor) -> Dict:
        """Select action with learned parameters"""
        with torch.no_grad():
            # Get action outputs
            output = self.actor(state.unsqueeze(0) if len(state.shape) == 1 else state)
            action_probs = output['action_probs']
            
            # Apply action mask
            masked_probs = action_probs * action_mask.unsqueeze(0) if len(state.shape) == 1 else action_probs * action_mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Sample action type
            action_dist = torch.distributions.Categorical(masked_probs)
            action_type_idx = action_dist.sample()
            action_type = self.action_types[action_type_idx.item()]
            
            # Get value estimate
            value = self.critic(state.unsqueeze(0) if len(state.shape) == 1 else state)
            
            # Get corresponding parameters
            parameters = self._get_action_parameters(action_type, output['parameters'][action_type])
            
            return {
                'action_type': action_type,
                'action_idx': action_type_idx.item(),
                'parameters': parameters,
                'log_prob': action_dist.log_prob(action_type_idx),
                'value': value.squeeze()
            }
    
    def _get_action_parameters(self, action_type: str, param_output: Dict) -> Dict:
        """Convert parameter outputs to actual action parameters"""
        if action_type == 'split':
            return {}  # No parameters needed
        
        elif action_type == 'merge':
            # Sample isolation level from learned probabilities
            iso_probs = param_output['isolation_level_probs']
            iso_dist = torch.distributions.Categorical(iso_probs)
            isolation_level = iso_dist.sample().item()
            
            return {
                'isolation_level': isolation_level,
                'mu': param_output['mu_value'].item()
            }
        
        elif action_type == 'change_iso':
            # Sample target isolation level
            target_iso_probs = param_output['target_iso_probs']
            target_iso_dist = torch.distributions.Categorical(target_iso_probs)
            target_isolation = target_iso_dist.sample().item()
            
            return {'target_isolation': target_isolation}
        
        elif action_type == 'adjust_interval':
            return {'new_mu': param_output['new_mu'].item()}
        
        return {}
    
    def store_experience(self, state, action, reward, next_state, log_prob, value, action_mask):
        """Store experience in buffer"""
        self.buffer.add(state, action, reward, next_state, log_prob, value, action_mask)
    
    def update(self):
        """Update policy using PPO"""
        if len(self.buffer) < self.batch_size:
            return None, None
        
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.ppo_epochs):
            batch = self.buffer.sample(self.batch_size)
            if batch is None:
                continue
                
            # Convert to tensors
            states = torch.stack(batch['states'])
            actions = torch.tensor(batch['actions'], dtype=torch.long)
            rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
            next_states = torch.stack(batch['next_states'])
            old_log_probs = torch.stack(batch['log_probs'])
            old_values = torch.stack(batch['values'])
            action_masks = torch.stack(batch['action_masks'])
            
            # Calculate advantages
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze()
                targets = rewards + self.gamma * next_values
                advantages = targets - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update actor
            actor_output = self.actor(states)
            action_probs = actor_output['action_probs']
            
            # Apply masks
            masked_probs = action_probs * action_masks
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # PPO loss
            action_dist = torch.distributions.Categorical(masked_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Update critic
            current_values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(current_values, targets)
            
            # Backpropagation
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # Clear buffer after update
        self.buffer.clear()
        
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        
        return avg_actor_loss, avg_critic_loss

class CompletePartitionAdjustmentSystem:
    """Complete system integrating all components"""
    
    def __init__(self, K: int = 3, lambda_param: float = 0.5):
        self.K = K
        self.lambda_param = lambda_param
        self.rl_agent = ParameterizedPPOAgent(state_dim=35)  # embedding(32) + 3 additional features
        self.performance_history = deque(maxlen=100)
        self.correctness_history = deque(maxlen=100)
        self.baseline_performance = None
        self.baseline_correctness = None
        self.time_step = 0
    
    def prepare_state_representation(self, partition: PartitionNode, embedding: torch.Tensor) -> torch.Tensor:
        """Prepare state vector for RL agent"""
        # State: [embedding(32), intensity, is_leaf, can_merge, isolation_level, mu]
        state_vector = torch.cat([
            embedding,
            torch.tensor([partition.workload_intensity]),
            torch.tensor([float(partition.is_leaf)]),
            torch.tensor([float(partition.can_merge)]),
            torch.tensor([partition.isolation_level]),
            torch.tensor([partition.mu])
        ])
        return state_vector
    
    def generate_action_mask(self, partition: PartitionNode) -> torch.Tensor:
        """Generate action mask based on partition constraints"""
        mask = torch.zeros(4)  # 4 action types
        
        if partition.is_leaf:
            mask[0] = 1  # split allowed
            mask[2] = 1  # change isolation allowed
            mask[3] = 1  # adjust interval allowed
            
        if partition.can_merge:
            mask[1] = 1  # merge allowed
            
        return mask
    
    def execute_action(self, partition: Partition, action: Dict) -> Dict:
        """Execute action with learned parameters"""
        action_type = action['action_type']
        parameters = action['parameters']
        
        if action_type == 'split' and partition.is_leaf:
            return self.execute_split(partition)
        elif action_type == 'merge' and partition.can_merge:
            return self.execute_merge(partition, parameters)
        elif action_type == 'change_iso' and partition.is_leaf:
            return self.execute_change_isolation(partition, parameters)
        elif action_type == 'adjust_interval' and partition.is_leaf:
            return self.execute_adjust_interval(partition, parameters)
        else:
            return {'success': False, 'error': 'Invalid action for partition state'}
    
    def execute_split(self, partition: Partition) -> Dict:
        """Execute split action"""
        # Create child partitions
        left_child = Partition(partition.partition_id * 2, partition.isolation_level)
        right_child = Partition(partition.partition_id * 2 + 1, partition.isolation_level)
        
        # Update partition
        partition.is_leaf = False
        partition.can_merge = True
        partition.left = left_child
        partition.right = right_child
        left_child.parent = partition
        right_child.parent = partition
        
        return {
            'success': True,
            'action': 'split',
            'left_child_id': left_child.partition_id,
            'right_child_id': right_child.partition_id
        }
    
    def execute_merge(self, partition: Partition, parameters: Dict) -> Dict:
        """Execute merge action with learned parameters"""
        if not (partition.left and partition.right):
            return {'success': False, 'error': 'No children to merge'}
        
        # Use learned parameters
        isolation_level = parameters.get('isolation_level', partition.isolation_level)
        mu = parameters.get('mu', partition.mu)
        
        # Update partition
        partition.isolation_level = isolation_level
        partition.mu = mu
        partition.is_leaf = True
        partition.can_merge = False
        partition.left = None
        partition.right = None
        
        return {
            'success': True,
            'action': 'merge',
            'isolation_level': isolation_level,
            'mu': mu
        }
    
    def calculate_reward(self, current_performance: float, current_correctness: float,
                       alpha: float = 0.7, eta_p: float = 0.5, eta_c: float = 0.5) -> Tuple[float, float, float]:
        """Calculate reward based on design document formula"""
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            self.baseline_correctness = current_correctness
            return 0.0, 0.0, 0.0
        
        previous_performance = self.performance_history[-1] if self.performance_history else self.baseline_performance
        previous_correctness = self.correctness_history[-1] if self.correctness_history else self.baseline_correctness
        
        # Performance improvement
        term1_p = (current_performance - self.baseline_performance) / self.baseline_performance
        term2_p = (current_performance - previous_performance) / previous_performance if previous_performance > 0 else 0
        R_p = eta_p * term1_p + (1 - eta_p) * term2_p
        
        # Correctness penalty
        term1_c = (current_correctness - self.baseline_correctness) / self.baseline_correctness
        term2_c = (current_correctness - previous_correctness) / previous_correctness if previous_correctness > 0 else 0
        P_c = eta_c * term1_c + (1 - eta_c) * term2_c
        
        # Combined reward
        reward = alpha * R_p - (1 - alpha) * P_c
        
        return reward, R_p, P_c
    
    def run_complete_cycle(self, partitions: List[PartitionNode], embeddings: Dict[int, torch.Tensor],
                         current_performance: float, current_correctness: float) -> Dict:
        """Run one complete adjustment cycle"""
        self.time_step += 1
        
        # Select top-K partitions (simplified selection)
        selected_partitions = partitions[:self.K]
        
        # Execute actions for all selected partitions
        adjustment_results = {}
        experiences = []
        
        for partition in selected_partitions:
            if partition.p_id not in embeddings:
                continue
                
            # Prepare RL inputs
            state = self.prepare_state_representation(partition, embeddings[partition.p_id])
            action_mask = self.generate_action_mask(partition)
            
            # Select action
            action_result = self.rl_agent.select_parameterized_action(state, action_mask)
            
            # Execute action
            result = self.execute_action(partition, action_result)
            adjustment_results[partition.p_id] = result
            
            # Calculate reward (simplified - in practice would be based on actual outcomes)
            reward = 0.1 if result['success'] else -0.1
            
            # Store experience
            next_state = self.prepare_state_representation(partition, embeddings[partition.p_id])
            self.rl_agent.store_experience(
                state, action_result['action_idx'], reward, next_state,
                action_result['log_prob'], action_result['value'], action_mask
            )
        
        # Calculate overall reward
        overall_reward, R_p, P_c = self.calculate_reward(current_performance, current_correctness)
        
        # Update performance history
        self.performance_history.append(current_performance)
        self.correctness_history.append(current_correctness)
        
        # Update RL policy
        actor_loss, critic_loss = self.rl_agent.update()
        
        return {
            'selected_partitions': [p.partition_id for p in selected_partitions],
            'adjustment_results': adjustment_results,
            'overall_reward': overall_reward,
            'performance_improvement': R_p,
            'correctness_penalty': P_c,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'time_step': self.time_step
        }

# Example usage
def example_usage():
    """Demonstrate the complete system"""
    system = CompletePartitionAdjustmentSystem(K=2)
    
    # Create mock partitions
    partitions = [
        Partition(1, isolation_level=0, is_leaf=True),
        Partition(2, isolation_level=1, is_leaf=True),
        Partition(3, isolation_level=2, is_leaf=False),
        Partition(4, isolation_level=0, is_leaf=True)
    ]
    
    # Set workload intensities
    for i, p in enumerate(partitions):
        p.workload_intensity = 0.1 * (i + 1)
        if i == 2:  # Make one partition mergeable
            p.can_merge = True
            p.left = Partition(5)
            p.right = Partition(6)
    
    # Mock embeddings
    embeddings = {p.partition_id: torch.randn(32) for p in partitions}
    
    # Mock performance metrics
    current_performance = 1000.0
    current_correctness = 5.0
    
    # Run adjustment cycle
    results = system.run_complete_cycle(partitions, embeddings, current_performance, current_correctness)
    
    print("Adjustment Cycle Results:")
    print(f"Selected partitions: {results['selected_partitions']}")
    print(f"Overall reward: {results['overall_reward']:.4f}")
    print(f"Actor loss: {results.get('actor_loss', 'N/A')}")
    print(f"Critic loss: {results.get('critic_loss', 'N/A')}")
    
    for pid, result in results['adjustment_results'].items():
        print(f"Partition {pid}: {result['action']} - {'Success' if result['success'] else 'Failed'}")

if __name__ == "__main__":
    example_usage()