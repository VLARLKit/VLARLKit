import torch
import torch.nn as nn


class TanhTransform(torch.distributions.Transform):
    """Tanh bijective transform with log_abs_det_jacobian."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self):
        super().__init__(cache_size=1)

    def __call__(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.clamp(y, -0.999999, 0.999999)
        return 0.5 * torch.log((1 + y) / (1 - y))

    def log_abs_det_jacobian(self, x, y):
        return torch.sum(torch.log(1 - y.pow(2) + 1e-7), dim=-1)


class GaussianPolicy(nn.Module):
    """Gaussian policy for SAC actor.

    Input: feature vector -> SquashedNormal -> (actions, log_prob)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=(128, 128, 128),
        log_std_init=-2.0,
        low=None,
        high=None,
        action_horizon=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._action_horizon = action_horizon
        self._low = low
        self._high = high

        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim

        self._shared_net = nn.Sequential(*layers)
        self._mean_layer = nn.Linear(in_dim, output_dim)
        self._log_std_layer = nn.Linear(in_dim, output_dim)

        self._init_weights(log_std_init)

    def _init_weights(self, log_std_init):
        nn.init.xavier_uniform_(self._mean_layer.weight, gain=0.01)
        nn.init.zeros_(self._mean_layer.bias)
        nn.init.xavier_uniform_(self._log_std_layer.weight, gain=0.01)
        nn.init.zeros_(self._log_std_layer.bias)

    def forward(self, features):
        """Forward pass returning mean and clamped log_std.

        Args:
            features: [B, input_dim]
        Returns:
            mean: [B, output_dim]
            log_std: [B, output_dim] clamped to [-20, 2]
        """
        h = self._shared_net(features)
        mean = self._mean_layer(h)
        log_std = torch.clamp(self._log_std_layer(h), -20, 2)
        return mean, log_std

    def sample(self, features, deterministic=False):
        """Sample actions with CleanRL-style manual reparameterization.

        Args:
            features: [B, input_dim]
            deterministic: use tanh(mean) without noise
        Returns:
            action: [B, action_horizon, output_dim]
            log_prob: [B]
        """
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(features.shape[0], device=features.device)
        else:
            normal = torch.distributions.Normal(mean, std)
            base_dist = torch.distributions.Independent(normal, 1)

            x_t = base_dist.rsample()
            y_t = torch.tanh(x_t)

            log_prob = base_dist.log_prob(x_t)
            log_prob -= torch.sum(torch.log(1 - y_t.pow(2) + 1e-7), dim=-1)

            if self._low is not None and self._high is not None:
                scale_factor = (self._high - self._low) / 2.0
                shift = (self._high + self._low) / 2.0
                action = y_t * scale_factor + shift
                log_prob -= torch.sum(
                    torch.log(torch.abs(scale_factor) * torch.ones_like(y_t)), dim=-1
                )
            else:
                action = y_t

        # Expand to action_horizon: [B, output_dim] -> [B, action_horizon, output_dim]
        action = action.unsqueeze(1).repeat(1, self._action_horizon, 1)

        return action, log_prob
