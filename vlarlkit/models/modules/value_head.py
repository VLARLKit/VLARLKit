# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes=(512, 128),
        output_dim: int = 1,
        activation: str = "gelu",  # 'relu' or 'gelu'
        bias_last: bool = False,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        if activation.lower() == "relu":
            act = nn.ReLU
        elif activation.lower() == "gelu":
            act = nn.GELU
        elif activation.lower() == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))

        self.mlp = nn.Sequential(*layers)

        self._init_weights(activation.lower())

    def _init_weights(self, nonlinearity="relu"):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


def aggregate_q(q_values: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate Q-values across ensemble heads. q_values: [B, N]."""
    if mode == "min":
        return torch.min(q_values, dim=1, keepdim=True).values
    elif mode == "mean":
        return torch.mean(q_values, dim=1, keepdim=True)
    elif mode == "redq":
        n_heads = q_values.shape[1]
        idx = torch.randperm(n_heads, device=q_values.device)[:2]
        return torch.min(q_values[:, idx], dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown agg_q mode: {mode}")
