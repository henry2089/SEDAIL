import torch
from ddpm_disc.diffusion.diffusion import Diffusion
from ddpm_disc.diffusion.model import MLP
from ddpm_disc.diffusion.model import DualTimeMLP
from ddpm_disc.diffusion.model import NewMLP
import rlkit.torch.utils.pytorch_util as ptu


class DDPM_Disc(torch.nn.Module):

    def __init__(self, x_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim, disc_momentum, env_name='hopper', traj_num=1, trainer_version='new',
                 lr=0.0003, clamp_magnitude=10.0, classifier=None,ema_classifier=None, lambda_factor=0.001, t_stopgrad=2, extend_type='exp', cond_dim=16, condition_mlp=True,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 t_min=5,
                 t_max=30,
                 is_diff_adapt=False,
                 t_dim_mult=2,
                 t_dim=16,
                 x_emb_dim=128,
                 n_dim=32,
                 num_mid_layers=2,
                 x_emb_type='single',
                 x_dim_mult=2,
                 emb_type='Sequential',
                 drop_rate=0.2):
        super().__init__()
        self.classifier = classifier
        self.ema_classifier = ema_classifier
        self.extend_type = extend_type

        self.model = NewMLP(x_dim=x_dim, hid_dim=disc_hid_dim, device=device, t_dim=t_dim, n_dim=n_dim,
                            t_dim_mult=t_dim_mult, emb_type=emb_type, t_timesteps=n_timesteps, n_timesteps=t_max, drop_rate=drop_rate)
        self.diffusion = Diffusion(x_dim, action_dim, self.model, self.classifier, self.ema_classifier, max_action,
                                   beta_schedule=beta_schedule, traj_num=traj_num,
                                   n_timesteps=n_timesteps, clamp_magnitude=clamp_magnitude,
                                   lambda_factor=lambda_factor, t_stopgrad=t_stopgrad, env_name=env_name).to(device)

        self.diff_opti = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=(disc_momentum, 0.999))


    def disc_reward(self, x):
        batch = x.to(ptu.device)
        disc_cost = self.diffusion.calc_reward(batch)
        return disc_cost

    def disc_condition_reward(self, x, offp_weight=None, step_loop=0, change_lambda_factor=False, epoch=0):
        batch = x.to(ptu.device)
        disc_cost = self.diffusion.calc_condition_reward(batch, offp_weight, self.extend_type, step_loop, change_lambda_factor, epoch)
        return disc_cost