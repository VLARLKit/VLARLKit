"""Independent action-conditioned Q heads for fixed-reference policy evaluation."""
import torch
from torch import nn


class _ActionInjectedQ(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dims):
        super().__init__()
        self._layers = nn.ModuleList()
        self._norms = nn.ModuleList()
        for width in hidden_dims:
            self._layers.append(nn.Linear(feature_dim + action_dim, width))
            self._norms.append(nn.LayerNorm(width))
            feature_dim = width
        # A scalar LayerNorm would erase the value, so the readout has no norm.
        self._output = nn.Linear(feature_dim + action_dim, 1)

    def forward(self, features, actions):
        hidden = features
        for layer, norm in zip(self._layers, self._norms):
            hidden = torch.relu(norm(layer(torch.cat((hidden, actions), dim=-1))))
        return self._output(torch.cat((hidden, actions), dim=-1))


class ActionInjectedEnsembleQ(nn.Module):
    def __init__(self, feature_dim=2056, action_dim=70,
                 hidden_dims=(512, 512, 512), heads=10):
        super().__init__()
        self._heads = nn.ModuleList([
            _ActionInjectedQ(feature_dim, action_dim, hidden_dims)
            for _ in range(heads)
        ])

    def forward(self, features, actions):
        features, actions = features.float(), actions.float().flatten(1)
        return torch.cat([head(features, actions) for head in self._heads], dim=-1)


def reference_td_targets(rewards, discounts, terminations, next_q, lcb_coefficient=.5,
                         target_min=-1000., target_max=0.):
    """Clamp the shared n-step target to the reward -1, gamma .999 return range."""
    lcb = next_q.mean(-1) - lcb_coefficient * next_q.std(-1, unbiased=False)
    assert next_q.ndim == 3 and next_q.shape[1:] == (8, 10)
    best_lcb = lcb.max(dim=1).values
    continuation = torch.where(terminations, torch.zeros_like(best_lcb), best_lcb)
    scalar_targets = (rewards + discounts * continuation).clamp(target_min, target_max)
    return scalar_targets[:, None].expand(-1, next_q.shape[-1])


def reference_td_loss(q, target_q, features, actions, next_features,
                      next_actions, rewards, discounts, terminations):
    with torch.no_grad():
        batch, candidates = next_actions.shape[:2]
        expanded_features = next_features[:, None].expand(-1, candidates, -1).reshape(batch*candidates, -1)
        next_estimates = target_q(expanded_features, next_actions.flatten(0, 1)).reshape(batch, candidates, -1)
        targets = reference_td_targets(
            rewards, discounts, terminations, next_estimates)
    predictions = q(features, actions)
    errors = (predictions - targets).square()
    # Independent heads: summing their losses preserves each head's original LR.
    return errors.mean(0).sum(), predictions, targets, next_estimates
