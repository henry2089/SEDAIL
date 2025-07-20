import numpy as np
import torch
import torch.nn as nn

from ddpm_disc.diffusion.helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
)
from ddpm_disc.diffusion.utils import (
    Progress,
    Silent,
    normal_kl,
    discretized_gaussian_log_likelihood,
    mean_flat,
)


class Diffusion(nn.Module):
    def __init__(
        self,
        x_dim,
        action_dim,
        model,
        classifier,
        ema_classifier,
        max_action,
        beta_schedule='linear',
        n_timesteps=100,
        loss_type='l2',
        clip_denoised=True,
        traj_num=1,
        predict_epsilon=True,
        clamp_magnitude=10.0,
        temperature=5.0,
        lambda_factor=0.001,
        n_guide_steps=1,
        t_stopgrad=2,
        env_name='hopper',
    ):
        super(Diffusion, self).__init__()

        self.traj_num = traj_num
        self.env_name = env_name
        self.lambda_factor = lambda_factor           # Base guidance strength
        self.lambda_decay_rate = 0.2                 # Time-step decay strength (reserved)
        self.lambda_max = 100.0                      # Safety upper bound (reserved)
        self.grad_momentum = 0.9                     # Momentum coefficient (reserved)
        self.x_dim = x_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.classifier = classifier                 # Main classifier
        self.ema_classifier = ema_classifier         # EMA classifier (frozen)
        self.temperature = temperature
        self.n_guide_steps = n_guide_steps
        self.t_stopgrad = t_stopgrad

        self.clamp_magnitude = clamp_magnitude
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Forward diffusion buffers
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        If self.predict_epsilon: model output is (scaled) noise.
        Otherwise, model directly predicts x0.
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t, is_dict=False):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if not is_dict:
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        else:
            posterior_variance = extract(self.posterior_variance, t, x_t.shape) * torch.ones_like(x_t)
            posterior_log_variance_clipped = (
                extract(self.posterior_log_variance_clipped, t, x_t.shape) * torch.ones_like(x_t)
            )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, is_dict=False):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))

        if self.clip_denoised:
            x_recon[:, :self.action_dim].clamp_(-self.max_action, self.max_action)
        else:
            raise RuntimeError("Clipping disabled without alternative handling.")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t, is_dict=is_dict
        )
        if not is_dict:
            return model_mean, posterior_variance, posterior_log_variance
        else:
            return {
                "mean": model_mean,
                "variance": posterior_variance,
                "log_variance": posterior_log_variance,
                "pred_xstart": x_recon,
            }

    def p_sample(self, x, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps)
            progress.update({'t': i})
            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def sample(self, shape, *args, **kwargs):
        action_state = self.p_sample_loop(shape, *args, **kwargs)
        action_state[:, :self.action_dim] = action_state[:, :self.action_dim].clamp_(
            -self.max_action, self.max_action
        )
        return action_state

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, t, weights=1.0, disc_ddpm=False):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            if not disc_ddpm:
                loss = self.loss_fn(x_recon, noise, weights)
            else:
                l2_ = torch.mean(torch.pow((x_recon - noise), 2), dim=1)
                loss = torch.exp(-l2_)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def loss(self, x, weights=1.0, disc_ddpm=False):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, weights, disc_ddpm)

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def calc_reward(self, x_start):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        for t in list(range(0, self.n_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                x_recon = self.model(x_t, t_batch)
                l2_ = torch.pow((x_recon - noise), 2)
                loss_t = torch.exp(-l2_)
            vb.append(loss_t)

        vb = torch.stack(vb, dim=1)
        disc_cost1 = vb.sum(dim=1) / self.n_timesteps
        disc_cost1 = disc_cost1.sum(dim=1) / x_start.shape[1]
        return disc_cost1

    def calc_condition_reward(
        self,
        x_start,
        offp_weight=None,
        extend_type='abs',
        step_loop=0,
        change_lambda_factor=False,
        epoch=0,
    ):
        device = x_start.device
        batch_size = x_start.shape[0]
        vb = []
        for t in list(range(0, self.n_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                x_recon = self.model(x_t, t_batch)

            x_t_grad = x_t.clone().detach().requires_grad_(True)  # Compute gradient only w.r.t x_t_grad

            for i in range(self.n_guide_steps):
                with torch.enable_grad():
                    ema_pred = self.ema_classifier(x_t_grad, t_batch)  # This classifier is frozen
                    ema_pred = ema_pred.squeeze(dim=-1)
                    normalized_weight = ema_pred

                    if extend_type == 'abs':
                        epsilon = 1e-10
                        w_diff = normalized_weight - 1
                        y = torch.sqrt(w_diff ** 2 + epsilon)
                    elif extend_type == 'exp':
                        e = torch.pow((normalized_weight - 1), 2)
                        y = -torch.exp(-e)
                    elif extend_type == 'sq':
                        w_diff = normalized_weight - 1
                        y = torch.pow(w_diff, 2)
                    else:
                        raise ValueError(f"Unknown extend_type: {extend_type}")

                    grad = torch.autograd.grad(y.sum(), [x_t_grad])[0]
                    grad = torch.clamp(grad, min=-30.0, max=30.0)  # Symmetric clipping

                if t < self.t_stopgrad:
                    grad.zero_()

                sqrt_one_minus_alpha_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, t_batch, x_t_grad.shape
                )
                grad_adjusted = grad * sqrt_one_minus_alpha_t

                with torch.no_grad():
                    if step_loop == 100 and i == 0 and t == 3:
                        print(f"x_recon before adjustment: {x_recon.mean()}")
                        print(f"grad_adjusted: {grad_adjusted.mean()}")
                    x_recon = x_recon + self.lambda_factor * grad_adjusted

            with torch.no_grad():
                l2_ = torch.pow((x_recon - noise), 2)
                loss_t = torch.exp(-l2_)
            vb.append(loss_t)

        vb = torch.stack(vb, dim=1)
        disc_cost1 = vb.sum(dim=1) / self.n_timesteps
        disc_cost1 = disc_cost1.sum(dim=1) / x_start.shape[1]
        return disc_cost1

    def classifier_p_losses(self, x_start, target, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred = self.classifier(x_noisy, t)
        loss = torch.mean(torch.pow((pred - target), 2), dim=1)
        return loss

    def classifier_forward(self, x, cond, t):
        return self.classifier(x, cond, t)

    def classifier_loss(self, x, target, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.classifier_p_losses(x, target, t, weights)

    def loss_chain(self, x, n_chain_steps, weights=1.0, disc_ddpm=False):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses_chain(x_start=x, t=t, n_chain_steps=n_chain_steps, weights=weights, disc_ddpm=disc_ddpm)

    def p_losses_chain(self, x_start, t, n_chain_steps, weights=1.0, disc_ddpm=False):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, n_chain_steps)

        if self.predict_epsilon:
            if not disc_ddpm:
                loss = self.loss_fn(x_recon, noise, weights)
            else:
                l2_ = torch.mean(torch.pow((x_recon - noise), 2), dim=1)
                loss = torch.exp(-l2_)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def reward_classifier(self, x_start, step_loop=0):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        for t in list(range(0, self.n_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            with torch.no_grad():
                x_recon = self.model(x_t, t_batch)

            for i in range(self.n_guide_steps):
                with torch.enable_grad():
                    x_in = x_t.detach().requires_grad_(True)
                    logits = self.ema_classifier(x_in, t_batch)  # This classifier is frozen
                    ema_pred = torch.sigmoid(logits).squeeze(dim=-1)
                    log_probs = torch.log(ema_pred)
                    grad = torch.autograd.grad(log_probs.sum(), [x_in])[0]

                if t < self.t_stopgrad:
                    grad.zero_()

                sqrt_one_minus_alpha_t = extract(
                    self.sqrt_one_minus_alphas_cumprod, t_batch, x_in.shape
                )
                grad_adjusted = grad * sqrt_one_minus_alpha_t

                with torch.no_grad():
                    if step_loop == 100 and i == 0 and t == 3:
                        print(f"x_recon before adjustment: {x_recon.mean()}")
                        print(f"grad_adjusted: {grad_adjusted.mean()}")
                    x_recon = x_recon - self.lambda_factor * grad_adjusted

            with torch.no_grad():
                l2_ = torch.pow((x_recon - noise), 2)
                loss_t = torch.exp(-l2_)
            vb.append(loss_t)

        vb = torch.stack(vb, dim=1)
        disc_cost1 = vb.sum(dim=1) / self.n_timesteps
        disc_cost1 = disc_cost1.sum(dim=1) / x_start.shape[1]
        return disc_cost1
