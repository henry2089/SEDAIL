import torch
import torch.nn as nn
from ddpm_disc.diffusion.helpers import SinusoidalPosEmb


class NewMLP(nn.Module):
    """
    Diffusion MLP supporting dual timesteps (t, n):
    - t: original diffusion process timestep
    - n: additional data augmentation noise timestep
    """
    def __init__(
        self,
        x_dim: int,
        hid_dim: int,
        device: torch.device,
        t_dim: int = 16,             # Sinusoidal embedding dim for t
        n_dim: int = 8,              # Sinusoidal embedding dim for n
        t_dim_mult: int = 2,
        drop_rate: float = 0.2,
        t_timesteps: int = 10,
        n_timesteps: int = 100,
        emb_type="Sequential",
    ):
        super().__init__()
        self.device = device
        self.emb_type = emb_type

        if self.emb_type == "Sequential":
            self.t_mlp = nn.Sequential(
                SinusoidalPosEmb(t_dim),
                nn.Linear(t_dim, t_dim * t_dim_mult),
                nn.Mish(),
                nn.Linear(t_dim * t_dim_mult, t_dim),
            )
            self.n_mlp = nn.Sequential(
                SinusoidalPosEmb(n_dim),
                nn.Linear(n_dim, n_dim * t_dim_mult),
                nn.Mish(),
                nn.Linear(n_dim * t_dim_mult, n_dim),
            )
        elif self.emb_type == "weight":
            self.time_weight = nn.Parameter(torch.ones(1))   # Learnable scaling factor for t
            self.noise_weight = nn.Parameter(torch.ones(1))  # Learnable scaling factor for n
            self.t_mlp = nn.Embedding(t_timesteps, t_dim)
            self.n_mlp = nn.Embedding(n_timesteps, n_dim)
        else:
            self.t_mlp = nn.Embedding(t_timesteps, t_dim)
            self.n_mlp = nn.Embedding(n_timesteps, n_dim)

        # Backbone MLP
        in_features = x_dim + t_dim + n_dim
        if drop_rate != 0.0:
            self.mid = nn.Sequential(
                nn.Linear(in_features, hid_dim), nn.Mish(),
                nn.Dropout(drop_rate),
                nn.Linear(hid_dim, hid_dim), nn.Mish(),
                nn.Dropout(drop_rate),
                nn.Linear(hid_dim, hid_dim), nn.Mish(),
            )
        else:
            self.mid = nn.Sequential(
                nn.Linear(in_features, hid_dim), nn.Mish(),
                nn.Linear(hid_dim, hid_dim), nn.Mish(),
                nn.Linear(hid_dim, hid_dim), nn.Mish(),
            )

        # Output layer: predict noise for x
        self.final = nn.Linear(hid_dim, x_dim)

    def forward(self, x, t, n=None):
        """
        x : (B, x_dim)
        t : (B,) diffusion timestep
        n : (B,) auxiliary noise timestep
        """
        if n is None:
            n = torch.zeros(x.size(0))
        x, t, n = x.to(self.device), t.to(self.device), n.to(self.device)

        if self.emb_type == "Sequential":
            t_emb = self.t_mlp(t)
            n_emb = self.n_mlp(n)
        elif self.emb_type == "weight":
            t_emb = self.time_weight * self.t_mlp(t.long())
            n_emb = self.noise_weight * self.n_mlp(n.long())
        else:
            t_emb = self.t_mlp(t.long())
            n_emb = self.n_mlp(n.long())

        x_emb = x
        h = torch.cat([x_emb, t_emb, n_emb], dim=-1)
        h = self.mid(h)
        return self.final(h)


class DualTimeMLP(nn.Module):
    """
    Diffusion MLP supporting dual timesteps (t, n):
    - t: original diffusion process timestep
    - n: additional augmentation noise timestep
    """
    def __init__(
        self,
        x_dim: int,
        hid_dim: int,
        device: torch.device,
        x_emb_dim: int = 128,     # Project x into this shared dimension
        t_dim: int = 16,
        n_dim: int = 8,
        t_dim_mult: int = 2,
        num_mid_layers: int = 3,
        x_emb_type='single',
        x_dim_mult=2,
    ):
        super().__init__()
        self.device = device
        self.x_emb_type = x_emb_type

        # x → shared feature dimension
        if x_emb_type == 'single':
            self.x_proj = nn.Sequential(
                nn.Linear(x_dim, x_emb_dim),
                nn.Mish(),
            )
        elif x_emb_type == 'multi':
            self.x_proj = nn.Sequential(
                nn.Linear(x_dim, x_emb_dim * x_dim_mult),
                nn.Mish(),
                nn.Linear(x_emb_dim * x_dim_mult, x_emb_dim),
            )
        elif x_emb_type == 'nemb':
            self.x_proj = nn.Sequential(
                nn.Linear(x_dim + n_dim, x_emb_dim * x_dim_mult),
                nn.Mish(),
                nn.Linear(x_emb_dim * x_dim_mult, x_emb_dim),
            )

        # t embedding
        self.t_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * t_dim_mult),
            nn.Mish(),
            nn.Linear(t_dim * t_dim_mult, t_dim),
        )

        # n embedding
        self.n_mlp = nn.Sequential(
            SinusoidalPosEmb(n_dim),
            nn.Linear(n_dim, n_dim * t_dim_mult),
            nn.Mish(),
            nn.Linear(n_dim * t_dim_mult, n_dim),
        )

        # Backbone MLP
        if x_emb_type == 'none':
            in_features = x_dim + t_dim + n_dim
        elif x_emb_type == 'nemb':
            in_features = x_emb_dim + t_dim
        else:
            in_features = x_emb_dim + t_dim + n_dim

        layers = []
        for _ in range(num_mid_layers):
            layers.append(nn.Linear(in_features, hid_dim))
            layers.append(nn.Mish())
            in_features = hid_dim
        self.mid = nn.Sequential(*layers)

        # Output layer: predict noise for x
        self.final = nn.Linear(hid_dim, x_dim)

    def forward(self, x, t, n=None):
        """
        x : (B, x_dim)
        t : (B,) diffusion timestep
        n : (B,) auxiliary noise timestep
        """
        if n is None:
            n = torch.zeros(x.size(0))
        x, t, n = x.to(self.device), t.to(self.device), n.to(self.device)

        t_emb = self.t_mlp(t)
        n_emb = self.n_mlp(n)

        if self.x_emb_type == 'none':
            x_emb = x
        elif self.x_emb_type == 'nemb':
            x = torch.cat([x, n_emb], dim=-1)
            x_emb = self.x_proj(x)
        else:
            x_emb = self.x_proj(x)

        if self.x_emb_type == 'nemb':
            h = torch.cat([x_emb, t_emb], dim=-1)
        else:
            h = torch.cat([x_emb, t_emb, n_emb], dim=-1)
        h = self.mid(h)
        return self.final(h)


class MLP(nn.Module):
    """Basic diffusion MLP with single timestep embedding."""
    def __init__(self, x_dim, hid_dim, device, t_dim=16, t_dim_mult=2):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * t_dim_mult),
            nn.Mish(),
            nn.Linear(t_dim * t_dim_mult, t_dim),
        )

        input_dim = x_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.Mish(),
            nn.Linear(hid_dim, hid_dim),
            nn.Mish(),
            nn.Linear(hid_dim, hid_dim),
            nn.Mish(),
        )

        self.final_layer = nn.Linear(hid_dim, x_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = torch.cat([x, t], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)
