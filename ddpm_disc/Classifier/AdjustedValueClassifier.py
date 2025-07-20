import torch
import torch.nn as nn
from typing import List
from .utils import SUPPORTED_TIMESTEP_EMBEDDING


class AdjustedValueClassifier(nn.Module):
    """
    Combined value-style and classifier-style network adapted to flat transition input.
    Input shape: [batch_size, transition_dim] instead of [batch_size, horizon, transition_dim].
    """

    def __init__(
        self,
        input_size: int,
        emb_dim: int,
        hidden_dims: List[int],
        timestep_emb_type: str = "positional",
        output_size: int = 1,
    ):
        super().__init__()

        # Timestep embedding
        self.map_noise = SUPPORTED_TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim)

        # Project (observation + action) transition into embedding space
        self.transition_proj = nn.Linear(input_size, emb_dim)

        # MLP head: combines transition embedding and timestep embedding
        mlp_layers = [
            nn.Linear(2 * emb_dim, hidden_dims[0]),
            nn.SiLU(),
        ]
        for i in range(len(hidden_dims) - 1):
            mlp_layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.SiLU(),
            ]
        mlp_layers.append(nn.Linear(hidden_dims[-1], output_size))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, transition_dim) containing (obs + act).
            t: Tensor of shape (batch_size,) with diffusion timesteps.

        Returns:
            Tensor of shape (batch_size, output_size): predicted value / score.
        """
        x_emb = self.transition_proj(x)
        t_emb = self.map_noise(t)
        combined = torch.cat([x_emb, t_emb], dim=-1)
        return self.mlp(combined)
