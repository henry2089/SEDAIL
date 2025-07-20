import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm_disc.diffusion.helpers import SinusoidalPosEmb


class TimedClassifier_new(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=128,
        timestep_embed_dim=64,
        hidden_dim=256,
        timestep_emb_type: str = "positional",
    ):
        super(TimedClassifier_new, self).__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 2),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 2, timestep_embed_dim),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Network layers
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim + timestep_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Mish(),
        )

    def forward(self, x, timesteps):
        # Timestep embedding
        t = self.time_mlp(timesteps)
        x = torch.cat([x, t], dim=1)
        # Forward pass
        logits = self.mid_layer(x)
        return logits


class TimedClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        timestep_embed_dim=64,
        hidden_dim=256,
        num_layers=2,
        n_timesteps=10,
    ):
        super(TimedClassifier, self).__init__()

        self.timestep_embed = nn.Embedding(n_timesteps, timestep_embed_dim)  # Max timesteps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Network layers
        self.fc1 = nn.Linear(input_dim + timestep_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Binary classification output

    def forward(self, x, timesteps):
        # Timestep embedding
        timestep_embeds = self.timestep_embed(timesteps)
        # Concatenate input with timestep embedding
        x = torch.cat([x, timestep_embeds], dim=-1)
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
