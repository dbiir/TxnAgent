import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.func import functional_call
import math
import random


# ============================================================
# Multi-Head Parameterized Actor Network (4 action types)
# ============================================================

class MultiHeadParameterizedActor(nn.Module):
    """
    Multi-head actor that outputs:
      1) action-type logits  (4 types: split / merge / change_iso / adjust_interval)
      2) per-action-type parameter distributions
    """

    ACTION_TYPES = ['split', 'merge', 'change_iso', 'adjust_interval']

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()

        # Shared feature extractor (LayerNorm normalizes raw inputs)
        self.shared = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action-type head  (discrete: 4 logits)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)       # raw logits – NO softmax
        )

        # ---------- per-action parameter heads ----------

        # split:  iso_l (3 logits) + mu_l (1) + iso_r (3 logits) + mu_r (1) = 8
        self.split_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8)
        )

        # merge:  iso (3 logits) + mu (1) = 4
        self.merge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )

        # change_iso:  target iso (3 logits)
        self.change_iso_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # adjust_interval:  mu mean (1) – continuous
        self.adjust_interval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Learnable log-std for continuous mu heads
        self.log_std_mu = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------ #

    def forward(self, state):
        """
        Returns
        -------
        dict with keys:
          'action_logits' : (B, 4)    raw logits for action type
          'parameters'    : dict[str, dict]   per-action processed params
        """
        feat = self.shared(state)

        action_logits = self.action_type_head(feat)

        parameters = {}
        parameters['split'] = self._process_split(self.split_head(feat))
        parameters['merge'] = self._process_merge(self.merge_head(feat))
        parameters['change_iso'] = self._process_change_iso(self.change_iso_head(feat))
        parameters['adjust_interval'] = self._process_adjust_interval(self.adjust_interval_head(feat))

        return {
            'action_logits': action_logits,
            'parameters': parameters
        }

    # ------------------------------------------------------------------ #
    #  Parameter processing helpers (keep everything as tensors)
    # ------------------------------------------------------------------ #

    def _process_split(self, raw):
        # raw: (B, 8)  or  (8,)
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        return {
            'iso_l_logits': raw[:, :3],         # (B, 3)
            'mu_l_mean': raw[:, 3:4] * 10.0,    # (B, 1) scaled
            'iso_r_logits': raw[:, 4:7],         # (B, 3)
            'mu_r_mean': raw[:, 7:8] * 10.0      # (B, 1) scaled
        }

    def _process_merge(self, raw):
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        return {
            'iso_logits': raw[:, :3],            # (B, 3)
            'mu_mean': raw[:, 3:4] * 10.0        # (B, 1) scaled
        }

    def _process_change_iso(self, raw):
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        return {
            'iso_logits': raw                    # (B, 3)
        }

    def _process_adjust_interval(self, raw):
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        return {
            'mu_mean': torch.sigmoid(raw) * 10.0  # (B, 1) in [0, 10]
        }

    # ------------------------------------------------------------------ #
    #  Distribution helpers (used by MetaPPO for log-prob computation)
    # ------------------------------------------------------------------ #

    def get_distributions(self, state, action_mask=None, params=None):
        """
        Get all distributions needed for PPO loss.

        Parameters
        ----------
        state       : (B, state_dim)
        action_mask : (B, 4) or None – 1 = allowed, 0 = masked
        params      : dict for functional_call (MAML inner loop), or None

        Returns
        -------
        action_type_dist : Categorical over 4 types
        param_dists      : dict[str, dict of Distribution objects]
        """
        if params is None:
            out = self.forward(state)
        else:
            out = functional_call(self, params, (state,))

        action_logits = out['action_logits']

        # NaN guard: replace NaN logits with zeros (uniform fallback)
        if torch.isnan(action_logits).any():
            print("[WARN] NaN detected in action_logits, falling back to uniform", flush=True)
            action_logits = torch.zeros_like(action_logits)

        # Apply action mask (set masked logits to -inf)
        if action_mask is not None:
            action_logits = action_logits + (1 - action_mask) * (-1e8)

        action_type_dist = Categorical(logits=action_logits)

        mu_std = torch.nan_to_num(self.log_std_mu, nan=0.0).exp().clamp(min=1e-6)

        param_dists = {}
        p = out['parameters']

        def _safe(t):
            """Replace NaN/Inf with zeros to prevent distribution errors."""
            return torch.where(torch.isfinite(t), t, torch.zeros_like(t))

        param_dists['split'] = {
            'iso_l': Categorical(logits=_safe(p['split']['iso_l_logits'])),
            'mu_l': Normal(_safe(p['split']['mu_l_mean']), mu_std),
            'iso_r': Categorical(logits=_safe(p['split']['iso_r_logits'])),
            'mu_r': Normal(_safe(p['split']['mu_r_mean']), mu_std),
        }
        param_dists['merge'] = {
            'iso': Categorical(logits=_safe(p['merge']['iso_logits'])),
            'mu': Normal(_safe(p['merge']['mu_mean']), mu_std),
        }
        param_dists['change_iso'] = {
            'iso': Categorical(logits=_safe(p['change_iso']['iso_logits'])),
        }
        param_dists['adjust_interval'] = {
            'mu': Normal(_safe(p['adjust_interval']['mu_mean']), mu_std),
        }

        return action_type_dist, param_dists


# Keep a backward-compatible alias
Actor = MultiHeadParameterizedActor


# ============================================================
# Critic Network
# ============================================================

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# GAE
# ============================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t]
            + gamma * values[t+1] * (1 - dones[t])
            - values[t]
        )
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]

    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, torch.tensor(returns)


# ============================================================
# Helpers – per-action log-prob
# ============================================================

def _compute_param_log_prob(action_type_idx, param_dists, action_params):
    """
    Compute the log-probability of the *parameter* component for each
    sample in a batch.

    Distributions have batch shape (B, ...) from the full state batch.
    For each sample i we evaluate the i-th distribution at the i-th
    action parameter value.

    Parameters
    ----------
    action_type_idx : int or LongTensor (B,)
    param_dists     : dict[str, dict of Distribution]
    action_params   : dict  with keys  iso_l, mu_l, iso_r, mu_r, iso, mu
    """
    action_names = MultiHeadParameterizedActor.ACTION_TYPES

    if isinstance(action_type_idx, int):
        action_type_idx = torch.tensor([action_type_idx])

    B = action_type_idx.shape[0]
    log_probs_list = []

    for i in range(B):
        a = action_type_idx[i].item()
        name = action_names[a]

        lp = torch.tensor(0.0)
        dists = param_dists[name]

        # Each dist has batch shape (B, ...). We evaluate the full batch
        # at each param value (broadcasts), then pick element [i] to get
        # the log-prob from the i-th state's distribution.
        if name == 'split':
            lp = (
                dists['iso_l'].log_prob(action_params['iso_l'][i])[i]
                + dists['mu_l'].log_prob(action_params['mu_l'][i].unsqueeze(-1))[i].sum(-1)
                + dists['iso_r'].log_prob(action_params['iso_r'][i])[i]
                + dists['mu_r'].log_prob(action_params['mu_r'][i].unsqueeze(-1))[i].sum(-1)
            )
        elif name == 'merge':
            lp = (
                dists['iso'].log_prob(action_params['iso'][i])[i]
                + dists['mu'].log_prob(action_params['mu'][i].unsqueeze(-1))[i].sum(-1)
            )
        elif name == 'change_iso':
            lp = dists['iso'].log_prob(action_params['iso'][i])[i]
        elif name == 'adjust_interval':
            lp = dists['mu'].log_prob(action_params['mu'][i].unsqueeze(-1))[i].sum(-1)

        log_probs_list.append(lp)

    return torch.stack(log_probs_list)


# ============================================================
# Meta-PPO (MAML-style)
# ============================================================

class MetaPPO:
    def __init__(
        self,
        state_dim,
        inner_lr=1e-3,
        meta_lr=3e-4,
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01
    ):

        self.actor = MultiHeadParameterizedActor(state_dim)
        self.critic = Critic(state_dim)

        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=meta_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=meta_lr
        )

        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

    # --------------------------------------------------------
    # PPO Loss (4-action parameterized)
    # --------------------------------------------------------

    def ppo_loss(self, states, action_types, action_params,
                 action_masks, old_log_probs, advantages, params=None):
        """
        Parameters
        ----------
        states        : (B, state_dim)
        action_types  : LongTensor (B,)  – index in [0,3]
        action_params : dict of Tensors  – iso_l (B,), mu_l (B,1), etc.
        action_masks  : (B, 4)  – 1=allowed, 0=masked
        old_log_probs : (B,)
        advantages    : (B,)
        params        : optional adapted parameters (MAML)
        """
        action_type_dist, param_dists = \
            self.actor.get_distributions(states, action_masks, params)

        # Action-type log-prob
        type_lp = action_type_dist.log_prob(action_types)

        # Parameter log-prob (per action type)
        param_lp = _compute_param_log_prob(action_types, param_dists, action_params)

        log_probs = type_lp + param_lp

        # Clamp log-ratio to prevent exp() overflow → inf → NaN weights
        log_ratio = (log_probs - old_log_probs).clamp(-20, 20)
        print("log_ratio: ", log_ratio)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_eps,
            1 + self.clip_eps
        ) * advantages

        # Entropy bonus (action-type entropy only for stability)
        entropy = action_type_dist.entropy()

        loss = -torch.min(surr1, surr2).mean()
        loss -= self.entropy_coef * entropy.mean()

        return loss

    # --------------------------------------------------------
    # Inner Adaptation
    # --------------------------------------------------------

    def inner_update(self, task_data):

        states, action_types, action_params, action_masks, \
            rewards, dones, old_log_probs = task_data

        values = self.critic(states).detach().tolist()
        advantages, returns = compute_gae(
            rewards, values, dones,
            self.gamma, self.lam
        )

        loss = self.ppo_loss(
            states, action_types, action_params,
            action_masks, old_log_probs, advantages
        )

        grads = torch.autograd.grad(
            loss,
            self.actor.parameters(),
            create_graph=True,
            allow_unused=True        # not all param heads used every batch
        )

        adapted_params = {
            name: param - self.inner_lr * grad if grad is not None else param
            for (name, param), grad in zip(
                self.actor.named_parameters(),
                grads
            )
        }

        return adapted_params

    # --------------------------------------------------------
    # Meta Update
    # --------------------------------------------------------

    def meta_update(self, task_batch):

        meta_loss = 0

        for support_data, query_data in task_batch:

            adapted_params = self.inner_update(support_data)

            states, action_types, action_params, action_masks, \
                rewards, dones, old_log_probs = query_data

            values = self.critic(states).detach().tolist()
            advantages, returns = compute_gae(
                rewards, values, dones,
                self.gamma, self.lam
            )

            loss = self.ppo_loss(
                states, action_types, action_params,
                action_masks, old_log_probs, advantages,
                params=adapted_params
            )

            meta_loss += loss

        meta_loss /= len(task_batch)

        # Skip update if loss is inf/NaN to prevent weight corruption
        loss_val = meta_loss.item()
        if not math.isfinite(loss_val):
            print(f"[WARN] meta_loss={loss_val}, skipping update", flush=True)
            return loss_val

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.meta_optimizer.step()

        return loss_val

    # --------------------------------------------------------
    # Critic Update (standard shared critic)
    # --------------------------------------------------------

    def update_critic(self, states, returns):

        values = self.critic(states)
        loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        return loss.item()

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "meta_opt": self.meta_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.meta_optimizer.load_state_dict(ckpt["meta_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])


# ============================================================
# Dummy Rollout Collector (Example)
# Replace with your DB workload environment
# ============================================================

def collect_dummy_rollout(state_dim, horizon=10):

    states = []
    action_types_list = []
    iso_l_list, mu_l_list = [], []
    iso_r_list, mu_r_list = [], []
    iso_list, mu_list = [], []
    action_masks_list = []
    rewards = []
    dones = []
    old_log_probs = []

    actor = MultiHeadParameterizedActor(state_dim)
    action_names = MultiHeadParameterizedActor.ACTION_TYPES

    state = torch.randn(state_dim)

    for t in range(horizon):
        # Random action mask (simulating partition constraints)
        action_mask = torch.zeros(4)
        # Always allow at least one action
        action_mask[0] = 1  # split
        action_mask[2] = 1  # change_iso
        action_mask[3] = 1  # adjust_interval
        if random.random() > 0.5:
            action_mask[1] = 1  # merge sometimes allowed

        action_type_dist, param_dists = actor.get_distributions(
            state.unsqueeze(0), action_mask.unsqueeze(0)
        )

        action_type = action_type_dist.sample()          # (1,)
        action_idx = action_type.item()
        action_name = action_names[action_idx]

        # Sample parameters from the chosen action's distributions
        iso_l = param_dists['split']['iso_l'].sample().squeeze(0)   # ()
        mu_l = param_dists['split']['mu_l'].sample().squeeze(0)     # (1,)
        iso_r = param_dists['split']['iso_r'].sample().squeeze(0)
        mu_r = param_dists['split']['mu_r'].sample().squeeze(0)
        iso = param_dists['merge']['iso'].sample().squeeze(0) if action_name in ('merge', 'change_iso') \
              else param_dists['change_iso']['iso'].sample().squeeze(0)
        mu = param_dists['merge']['mu'].sample().squeeze(0) if action_name in ('merge', 'adjust_interval') \
             else param_dists['adjust_interval']['mu'].sample().squeeze(0)

        # Compute joint log-prob
        type_lp = action_type_dist.log_prob(action_type).squeeze(0)
        param_lp_val = torch.tensor(0.0)
        if action_name == 'split':
            param_lp_val = (
                param_dists['split']['iso_l'].log_prob(iso_l)
                + param_dists['split']['mu_l'].log_prob(mu_l).sum(-1)
                + param_dists['split']['iso_r'].log_prob(iso_r)
                + param_dists['split']['mu_r'].log_prob(mu_r).sum(-1)
            ).squeeze(0)
        elif action_name == 'merge':
            param_lp_val = (
                param_dists['merge']['iso'].log_prob(iso)
                + param_dists['merge']['mu'].log_prob(mu).sum(-1)
            ).squeeze(0)
        elif action_name == 'change_iso':
            param_lp_val = param_dists['change_iso']['iso'].log_prob(iso).squeeze(0)
        elif action_name == 'adjust_interval':
            param_lp_val = param_dists['adjust_interval']['mu'].log_prob(mu).sum(-1).squeeze(0)

        log_prob = type_lp + param_lp_val

        next_state = torch.randn(state_dim)
        reward = random.random()
        done = 0

        states.append(state)
        action_types_list.append(action_type.squeeze(0))
        iso_l_list.append(iso_l)
        mu_l_list.append(mu_l)
        iso_r_list.append(iso_r)
        mu_r_list.append(mu_r)
        iso_list.append(iso)
        mu_list.append(mu)
        action_masks_list.append(action_mask)
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(log_prob.detach())

        state = next_state

    action_params = {
        'iso_l': torch.stack(iso_l_list),
        'mu_l': torch.stack(mu_l_list),
        'iso_r': torch.stack(iso_r_list),
        'mu_r': torch.stack(mu_r_list),
        'iso': torch.stack(iso_list),
        'mu': torch.stack(mu_list),
    }

    return (
        torch.stack(states),           # (H, state_dim)
        torch.stack(action_types_list),  # (H,)
        action_params,                 # dict of (H, ...) tensors
        torch.stack(action_masks_list),  # (H, 4)
        rewards,                       # list
        dones,                         # list
        torch.stack(old_log_probs)     # (H,)
    )


# ============================================================
# Training Entrance
# ============================================================

if __name__ == "__main__":

    state_dim = 36
    agent = MetaPPO(state_dim)

    for meta_iter in range(100):

        task_batch = []

        for _ in range(5):  # 5 tasks per meta batch

            support = collect_dummy_rollout(state_dim)
            query = collect_dummy_rollout(state_dim)

            task_batch.append((support, query))

        meta_loss = agent.meta_update(task_batch)

        print(f"Meta Iter {meta_iter} | Meta Loss: {meta_loss:.4f}")

    agent.save("meta_ppo.pt")
    print("Model saved.")
