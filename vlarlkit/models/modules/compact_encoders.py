import math

import torch
import torch.nn as nn


class LightweightImageEncoder64(nn.Module):
    """Lightweight CNN encoder for 64x64 images.

    4 Conv layers (32 filters each), stride (2,1,1,1) -> bottleneck -> latent_dim.
    ~13K params per encoder.
    """

    def __init__(self, num_images=1, latent_dim=64, image_size=64):
        super().__init__()
        self.num_images = num_images
        self.latent_dim = latent_dim
        self.image_size = image_size

        final_h = image_size // 2
        final_w = image_size // 2
        self._flat_dim = 32 * final_h * final_w

        self.encoder = nn.Sequential(
            nn.Conv2d(num_images * 3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images):
        """
        Args:
            images: [B, N, C, H, W] where N=num_images (H=W=64)
        Returns:
            features: [B, latent_dim]
        """
        B, N, C, H, W = images.shape
        x = images.view(B, N * C, H, W)

        weight_dtype = self.encoder[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)

        x = self.encoder(x)
        features = self.bottleneck(x)
        return features


class CompactStateEncoder(nn.Module):
    """Compact state encoder: Linear -> LayerNorm -> Tanh."""

    def __init__(self, state_dim=32, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )

    def forward(self, state):
        """
        Args:
            state: [B, state_dim]
        Returns:
            features: [B, hidden_dim]
        """
        if state.dim() > 2:
            state = state.reshape(state.shape[0], -1)

        weight_dtype = self.encoder[0].weight.dtype
        if state.dtype != weight_dtype:
            state = state.to(dtype=weight_dtype)

        return self.encoder(state)


class CompactQHead(nn.Module):
    """Single Q-head MLP: concat(state, image, action) -> Q-value."""

    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),
        output_dim=1,
    ):
        super().__init__()

        input_dim = state_dim + image_dim + action_dim

        layers = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.extend(
                [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for i, m in enumerate(self.net):
            if isinstance(m, nn.Linear):
                if i == len(self.net) - 1:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features: [B, state_dim]
            image_features: [B, image_dim]
            actions: [B, action_dim]
        Returns:
            q_values: [B, 1]
        """
        x = torch.cat([state_features, image_features, actions], dim=-1)
        weight_dtype = self.net[0].weight.dtype
        if x.dtype != weight_dtype:
            x = x.to(dtype=weight_dtype)
        return self.net(x)


class CompactMultiQHead(nn.Module):
    """Ensemble of CompactQHead networks."""

    def __init__(
        self,
        state_dim=64,
        image_dim=64,
        action_dim=32,
        hidden_dims=(128, 128, 128),
        num_q_heads=10,
        output_dim=1,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads

        self.q_heads = nn.ModuleList(
            [
                CompactQHead(state_dim, image_dim, action_dim, hidden_dims, output_dim)
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features: [B, state_dim]
            image_features: [B, image_dim]
            actions: [B, action_dim]
        Returns:
            q_values: [B, num_q_heads]
        """
        q_values = []
        for q_head in self.q_heads:
            q_values.append(q_head(state_features, image_features, actions))
        return torch.cat(q_values, dim=-1)
