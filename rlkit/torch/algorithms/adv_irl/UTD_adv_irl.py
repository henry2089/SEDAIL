import numpy as np
from collections import OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from .UTD_torch_base_algorithm import UTDTorchBaseAlgorithm
from rlkit.torch.algorithms.adv_irl.utility.bypass_bn import enable_running_stats, disable_running_stats
import copy

from torch.optim import Optimizer


class CentralizedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # Preserve original Adam configuration
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step with gradient centralization"""
        loss = None
        if closure is not None:
            loss = closure()

        # ===== Core logic of Gradient Centralization =====
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                # Handle parameters of different dimensions:
                # Conv weights (out_c, in_c, kH, kW): dims = 1,2,3
                # Linear weights (out_f, in_f): dim = 1
                if grad.ndim > 1:  # Only apply to multi-dimensional parameters
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.ndim)), keepdim=True))

        # ===== Original Adam optimization logic =====
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                state['step'] += 1
                state_steps.append(state['step'])

            # Call underlying Adam implementation
            optim.adam.adam(params_with_grad,
                            grads,
                            exp_avgs,
                            exp_avg_sqs,
                            max_exp_avg_sqs,
                            state_steps,
                            amsgrad=group['amsgrad'],
                            beta1=beta1,
                            beta2=beta2,
                            lr=group['lr'],
                            weight_decay=group['weight_decay'],
                            eps=group['eps'])

        return loss


class UTDAdvIRL(UTDTorchBaseAlgorithm):
    """
    Depending on choice of reward function and size of replay
    buffer this will be:
        - GAIL
        - SEDAIL
        - Discriminator Actor Critic

    I did not implement the reward-wrapping mentioned in
    https://arxiv.org/pdf/1809.02925.pdf though

    Features removed from v1.0:
        - gradient clipping
        - target disc (exponential moving average disc)
        - target policy (exponential moving average policy)
        - disc input noise
    """

    def __init__(
            self,
            mode,  # gail, or ddpm
            discriminator,
            policy_trainer,
            expert_replay_buffer,
            state_only=False,
            disc_optim_batch_size=1024,
            policy_optim_batch_size=1024,
            policy_optim_batch_size_from_expert=0,
            num_update_loops_per_train_call=1000,
            first_train_call_loops=1000,   # NOTE
            num_disc_updates_per_loop_iter=100,
            num_policy_updates_per_loop_iter=100,
            num_weight_per_train_call=2000,  # NOTE
            num_classifier_per_train_call=30000,
            first_weight=1000,
            first_classifier=10000,
            second_multi=2,
            gradient_accumulate_every=2,
            weight_net=None,
            weight_net_lr=3e-4,
            weight_optim_batch_size=256,
            disc_lr=1e-3,
            disc_momentum=0.0,
            disc_optimizer_class=optim.Adam,
            use_grad_pen=True,
            grad_pen_weight=10,
            rew_clip_min=None,
            rew_clip_max=None,
            disc_ddpm=False,

            ema_decay=0.999,  # EMA decay factor
            classifier_lr=2e-4,
            update_ema_every=10,
            trainer_type='REDQ',
            num_update=20000,
            num_update_disc=20000,
            disc_divisor=10,
            ada_const=1.0,
            ada_interval=4,
            ada_target=0.6,
            diffusion_chain=None,
            is_classifier_reward=True,
            eval_calc_batch_size=1024,
            num_eval_iterations=10,
            ischain=True,
            traj_num=1,
            train_type='same',
            reward_smooth_gamma=0.9,
            is_reward_smooth=True,
            train_divisor=10,
            is_clip=True,
            utd=20,
            update_type='all',
            clas_type='extent',
            clip_num=0.2,
            **kwargs
    ):
        assert mode in [
            "gail",
            "ddpm",
        ], "Invalid adversarial irl algorithm!"
        super().__init__(**kwargs)

        self.clip_num = clip_num
        self.clas_type = clas_type
        self.update_type = update_type
        self.utd = utd
        self.traj_num = traj_num

        self.is_clip = is_clip
        self.train_divisor = train_divisor
        self.train_type = train_type
        self.change_lambda_factor = False
        self.mode = mode
        self.disc_ddpm = disc_ddpm
        self.state_only = state_only

        self.expert_replay_buffer = expert_replay_buffer

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.num_weight_per_train_call = num_weight_per_train_call      # NOTE
        self.num_classifier_per_train_call = num_classifier_per_train_call
        self.first_weight = first_weight
        self.first_classifier = first_classifier
        self.second_multi = second_multi
        self.second_weight = second_multi * num_weight_per_train_call
        self.second_classifier = second_multi * num_classifier_per_train_call
        self.gradient_accumulate_every = gradient_accumulate_every
        self.ada_const = ada_const
        self.ada_interval = ada_interval
        self.ada_target = ada_target
        self.ischain = ischain
        # diffusion_chain
        self.diffchain = diffusion_chain
        # Initialize weight_net and related components
        self.weight_net = weight_net
        self.w_activation = lambda x: torch.relu(x)
        self.weight_optim_batch_size = weight_optim_batch_size
        self.is_classifier_reward = is_classifier_reward
        self.is_classifier_reward_eval = is_classifier_reward
        self.eval_calc_batch_size = eval_calc_batch_size
        self.num_eval_iterations = num_eval_iterations
        # Optimizer for weight_net
        self.weight_optimizer = optim.Adam(
            self.weight_net.parameters(), lr=weight_net_lr
        )

        # Reference classifier from discriminator
        self.discriminator = discriminator
        self.classifier = self.discriminator.classifier
        self.classifier_ema = self.discriminator.ema_classifier
        self.ema_decay = ema_decay
        self.update_ema_every = update_ema_every
        self.num_update = num_update
        self.num_update_disc = int(num_update_disc)
        self.disc_divisor = disc_divisor
        self.trainer_type = trainer_type

        # Initialize classifier optimizer
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), lr=classifier_lr
        )
        # Use centralized Adam (compatible with original hyperparameters)
        self.disc_optimizer = CentralizedAdam(
            self.discriminator.parameters(),
            lr=disc_lr,
            betas=(disc_momentum, 0.999),
            weight_decay=1e-4
        )

        self.disc_optim_batch_size = disc_optim_batch_size
        print("\n\nDISC MOMENTUM: %f\n\n" % disc_momentum)

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight * utd
        self.reward_ema = None
        self.reward_smooth_gamma = reward_smooth_gamma  # smoothing coefficient
        self.is_reward_smooth = is_reward_smooth  # enable reward smoothing

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.first_train_call_loops = first_train_call_loops
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None
        self.weight_eval_statistics = None
        self.reward_train_idx = 0

    def update_classifier_ema(self):
        """
        Update EMA classifier weights.
        """
        for ema_param, param in zip(self.classifier_ema.parameters(), self.classifier.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=(1 - self.ema_decay))

    def eval_mode(self, epoch):
        # Ensure eval mode during evaluation
        self.classifier.eval()
        self.classifier_ema.eval()

    def weight_training(self, epoch):
        """
        Importance Sampling weight network training.
        Sample from on-policy and off-policy buffers and train the weight network.
        """
        keys = ["observations", "actions"]
        # Sample from off-policy and on-policy buffers
        offpolicy_batch = self.get_batch(self.weight_optim_batch_size, 'offpolicy', keys)
        onpolicy_batch = self.get_batch(self.weight_optim_batch_size, 'onpolicy', keys)

        # Offline data (offpolicy buffer)
        offline_obs = offpolicy_batch["observations"]
        offline_actions = offpolicy_batch["actions"]
        offline_input = torch.cat([offline_actions, offline_obs], dim=1)
        # Online data (onpolicy buffer)
        online_obs = onpolicy_batch["observations"]
        online_actions = onpolicy_batch["actions"]
        online_input = torch.cat([online_actions, online_obs], dim=1)

        # Offline weights
        offline_weight = self.w_activation(self.weight_net(offline_input))
        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + 1e-10)

        # Online weights
        online_weight = self.w_activation(self.weight_net(online_input))
        online_f_prime = torch.log(2 * online_weight / (online_weight + 1) + 1e-10)

        # Weight net loss
        weight_loss = (offline_f_star - online_f_prime).mean()

        # Update weight network
        self.weight_optimizer.zero_grad()
        weight_loss.backward()
        self.weight_optimizer.step()

    def classifier_training(self, step, epoch):
        """
        Classifier training logic.
        """
        if self.clas_type == 'class':
            keys = ["observations", "actions"]
            offpolicy_batch = self.get_batch(self.weight_optim_batch_size, "offpolicy", keys)
            clas_obs_off = offpolicy_batch["observations"]
            clas_acts_off = offpolicy_batch["actions"]
            clas_input_off = torch.cat([clas_acts_off, clas_obs_off], dim=1)

            onpolicy_batch = self.get_batch(self.weight_optim_batch_size, "onpolicy", keys)
            clas_obs_on = onpolicy_batch["observations"]
            clas_acts_on = onpolicy_batch["actions"]
            clas_input_on = torch.cat([clas_acts_on, clas_obs_on], dim=1)

            # Sample diffusion timesteps
            t_batch = torch.randint(0, self.discriminator.diffusion.n_timesteps, (self.weight_optim_batch_size,), device=clas_input_off.device).long()

            # Generate noised samples
            noise = torch.randn_like(clas_input_off)
            x_t_off = self.discriminator.diffusion.q_sample(clas_input_off, t_batch, noise)
            x_t_on = self.discriminator.diffusion.q_sample(clas_input_on, t_batch, noise)

            # Compute losses
            logits_off = self.classifier(x_t_off, t_batch)
            logits_on = self.classifier(x_t_on, t_batch)

            loss_off = F.binary_cross_entropy_with_logits(logits_off, torch.zeros(logits_off.size(), device=ptu.device))
            loss_on = F.binary_cross_entropy_with_logits(logits_on, torch.ones(logits_on.size(), device=ptu.device))
            loss = loss_off + loss_on

            loss.backward()
        else:
            for i in range(self.gradient_accumulate_every):
                keys = ["observations", "actions"]
                batch = self.get_batch(self.weight_optim_batch_size, "offpolicy", keys)
                clas_obs = batch["observations"]
                clas_acts = batch["actions"]
                clas_input = torch.cat([clas_acts, clas_obs], dim=1)
                weight = self.w_activation(self.weight_net(clas_input))

                loss = self.discriminator.diffusion.classifier_loss(clas_input, target=weight)
                loss = loss.mean()
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            # Update classifier parameters
            self.classifier_optimizer.step()
            self.classifier_optimizer.zero_grad()

            # Update EMA classifier
            if step % self.update_ema_every == 0:
                self.update_classifier_ema()

    def sample_with_classifier(self, inputs):
        """
        Inference with EMA classifier.
        """
        with torch.no_grad():
            output = self.classifier_ema(inputs)
        return output

    def get_batch(self, batch_size, buffer_type='expert', keys=None):
        """
        Sample a batch from the specified buffer (expert, onpolicy, offpolicy).

        Args:
            batch_size: int, batch size to sample
            buffer_type: str, which buffer to sample from ('expert', 'onpolicy', 'offpolicy')
            keys: list, optional, which keys to sample

        Returns:
            batch: sampled data in PyTorch format
        """
        if buffer_type == 'expert':
            buffer = self.expert_replay_buffer
        elif buffer_type == 'onpolicy':
            buffer = self.onpolicy_replay_buffer
        elif buffer_type == 'offpolicy':
            buffer = self.replay_buffer
        else:
            raise ValueError(f"Invalid buffer_type: {buffer_type}. Choose from ['expert', 'onpolicy', 'offpolicy'].")

        batch = buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _end_epoch(self):
        self.policy_trainer.end_epoch()
        self.disc_eval_statistics = None
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()

        self.r_gap_overfit_calculate()
        self.eval_statistics.update(self.disc_eval_statistics)
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        super().evaluate(epoch)

    def r_gap_overfit_calculate(self):
        self.discriminator.diffusion.eval()

        disc_real_sum = 0.0
        disc_fake_sum = 0.0
        reward1_sum = 0.0
        reward1_sum_no = 0.0
        reward2_sum = 0.0
        reward2_sum_no = 0.0
        for i in range(self.num_eval_iterations):
            keys = ["observations", "actions"]
            if self.wrap_absorbing:
                keys.append("absorbing")
            batch = self.get_batch(self.eval_calc_batch_size, 'offpolicy', keys)
            obs = batch["observations"]
            acts = batch["actions"]
            if self.wrap_absorbing:
                obs = torch.cat([obs, batch["absorbing"][:, 0:1]], dim=-1)
            eval_input_replay = torch.cat([acts, obs], dim=1)

            # Reward gap calculation
            if self.is_classifier_reward_eval:  # NOTE
                self.discriminator.diffusion.eval()
                if self.clas_type == 'class':
                    replay_logits = self.discriminator.diffusion.reward_classifier(eval_input_replay).detach()
                else:
                    replay_logits = self.discriminator.disc_condition_reward(eval_input_replay).detach()
                self.discriminator.diffusion.train()
                r1 = replay_logits.to(ptu.device)
                self.discriminator.diffusion.eval()
                replay_logits_nocla = self.discriminator.disc_reward(eval_input_replay).detach()
                self.discriminator.diffusion.train()
                r1_no = replay_logits_nocla.to(ptu.device)
                reward1 = np.mean(ptu.get_numpy((- torch.log(1 - r1)).unsqueeze(dim=1)))
                reward1_no = np.mean(ptu.get_numpy((- torch.log(1 - r1_no)).unsqueeze(dim=1)))

                reward1_sum_no += reward1_no
            else:
                self.discriminator.diffusion.eval()
                replay_logits = self.discriminator.disc_reward(eval_input_replay).detach()
                self.discriminator.diffusion.train()
                r1 = replay_logits.to(ptu.device)
                reward1 = np.mean(ptu.get_numpy((- torch.log(1 - r1)).unsqueeze(dim=1)))

            with torch.no_grad():
                policy_outputs = self.policy_trainer.policy(obs, return_log_prob=True)
                new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
                eval_input_policy = torch.cat([new_actions, obs], dim=1)
            if self.is_classifier_reward_eval:  # NOTE
                self.discriminator.diffusion.eval()
                if self.clas_type == 'class':
                    policy_logits = self.discriminator.diffusion.reward_classifier(eval_input_policy).detach()
                else:
                    policy_logits = self.discriminator.disc_condition_reward(eval_input_policy).detach()
                self.discriminator.diffusion.train()
                r2 = policy_logits.to(ptu.device)
                self.discriminator.diffusion.eval()
                policy_logits_nocla = self.discriminator.disc_reward(eval_input_policy).detach()
                self.discriminator.diffusion.train()
                r2_no = policy_logits_nocla.to(ptu.device)
                reward2 = np.mean(ptu.get_numpy((- torch.log(1 - r2)).unsqueeze(dim=1)))
                reward2_no = np.mean(ptu.get_numpy((- torch.log(1 - r2_no)).unsqueeze(dim=1)))

                reward2_sum_no += reward2_no
            else:
                self.discriminator.diffusion.eval()
                policy_logits = self.discriminator.disc_reward(eval_input_policy).detach()
                self.discriminator.diffusion.train()
                r2 = policy_logits.to(ptu.device)
                reward2 = np.mean(ptu.get_numpy((- torch.log(1 - r2)).unsqueeze(dim=1)))

            reward1_sum += reward1
            reward2_sum += reward2

            # Overfit calculation
            expert_batch = self.get_batch(self.eval_calc_batch_size, 'expert', keys)
            expert_obs = expert_batch["observations"]
            expert_actions = expert_batch["actions"]
            expert_disc_input = torch.cat([expert_actions, expert_obs], dim=1)

            expert_d = self.discriminator.diffusion.loss(expert_disc_input, disc_ddpm=self.disc_ddpm).detach()
            d_real = expert_d.to(ptu.device)
            disc_real = np.mean(ptu.get_numpy(d_real))
            action_d = self.discriminator.diffusion.loss(eval_input_replay, disc_ddpm=self.disc_ddpm).detach()
            d_fake = action_d.to(ptu.device)
            disc_fake = np.mean(ptu.get_numpy(d_fake))
            disc_real_sum += disc_real
            disc_fake_sum += disc_fake
        r_gap = reward2_sum / self.num_eval_iterations - reward1_sum / self.num_eval_iterations
        r_gap_abs = abs(r_gap)
        if self.is_classifier_reward_eval:
            r_gap_no = reward2_sum_no / self.num_eval_iterations - reward1_sum_no / self.num_eval_iterations
            r_gap_no_abs = abs(r_gap_no)
            self.disc_eval_statistics["Reward Gap abs"] = r_gap_abs
            self.disc_eval_statistics["Reward Gap"] = r_gap
            self.disc_eval_statistics["Reward Gap noclassifier abs"] = r_gap_no_abs
            self.disc_eval_statistics["Reward Gap noclassifier"] = r_gap_no
        else:
            self.disc_eval_statistics["Reward Gap noclassifier abs"] = r_gap_abs
            self.disc_eval_statistics["Reward Gap noclassifier"] = r_gap

        disc_real = disc_real_sum / self.num_eval_iterations
        disc_fake = disc_fake_sum / self.num_eval_iterations

        self.discriminator.diffusion.train()
        self.disc_eval_statistics["disc_real"] = disc_real
        self.disc_eval_statistics["disc_fake"] = disc_fake
        self.disc_eval_statistics["diffchain_p"] = self.diffchain.p

    def _do_training(self, epoch):
        if self.is_classifier_reward_eval:
            if self._n_train_steps_total == 0:
                for step_loop in range(self.num_update_loops_per_train_call):
                    self._do_ddpm_reward_training(epoch)
                    self._do_policy_training(epoch, step_loop)
            elif self._n_train_steps_total == 1:
                if self.clas_type != 'class':
                    for _ in range(self.num_weight_per_train_call):
                        self.weight_training(epoch)
                for step in range(self.num_classifier_per_train_call):
                    self.classifier_training(epoch, step)
                for step_loop in range(self.num_update_loops_per_train_call):
                    self._do_ddpm_reward_training(epoch)
                    self._do_policy_training(epoch, step_loop)
            elif self.stop_UTD == False:
                if self.clas_type != 'class':
                    for _ in range(self.num_weight_per_train_call):
                        self.weight_training(epoch)
                for step in range(self.num_classifier_per_train_call):
                    self.classifier_training(epoch, step)
                for step_loop in range(self.num_update_loops_per_train_call * self.utd):
                    self._do_ddpm_reward_training(epoch)
                    self._do_policy_training(epoch, step_loop)
            else:
                if self.clas_type != 'class':
                    for _ in range(self.num_weight_per_train_call // 2):
                        self.weight_training(epoch)
                for step in range(self.num_classifier_per_train_call // 2):
                    self.classifier_training(epoch, step)
                for step_loop in range(self.num_update_loops_per_train_call):
                    self._do_ddpm_reward_training(epoch)
                    self._do_policy_training(epoch, step_loop)
        else:
            if "gail" in self.mode in self.mode:
                for _ in range(self.num_update):
                    self._do_reward_training(epoch)
                for step_loop in range(self.num_update):
                    self._do_policy_training(epoch, step_loop)
            elif "ddpm" in self.mode:
                for _ in range(self.num_update):
                    self._do_ddpm_reward_training(epoch)
                for step_loop in range(self.num_update):
                    self._do_policy_training(epoch, step_loop)

    def _do_ddpm_reward_training(self, epoch):
        """
        Train the discriminator
        """
        params_old = self.discriminator.model.parameters()
        params_old_list = []
        for param in params_old:
            params_old_list.append(param.data)

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        expert_batch = self.get_batch(self.disc_optim_batch_size, 'expert', keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, 'offpolicy', keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            policy_next_obs = policy_batch["next_observations"]
            if self.wrap_absorbing:
                expert_next_obs = torch.cat(
                    [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
                )
                policy_next_obs = torch.cat(
                    [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )
            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_acts, expert_obs], dim=1)
            policy_disc_input = torch.cat([policy_acts, policy_obs], dim=1)

        # Diffusion chain
        if self.ischain:
            expert_disc_input, time_expert = self.diffchain(expert_disc_input)
            policy_disc_input, time_policy = self.diffchain(policy_disc_input)

            expert_d = self.discriminator.diffusion.loss_chain(x=expert_disc_input, n_chain_steps=time_expert, disc_ddpm=self.disc_ddpm).unsqueeze(dim=1)
            actor_d = self.discriminator.diffusion.loss_chain(x=policy_disc_input, n_chain_steps=time_policy, disc_ddpm=self.disc_ddpm).unsqueeze(dim=1)

            if self.reward_train_idx % self.ada_interval == 0:
                ada_stats = expert_d.squeeze(dim=-1)
                adjusted_expert_d = ada_stats - 0.5
                sign_expert_d = adjusted_expert_d.sign()
                r_d = sign_expert_d.mean()
                adjust = np.sign((r_d - self.ada_target).detach().cpu().numpy()) * self.ada_const
                self.diffchain.p = (self.diffchain.p + adjust).clip(min=0., max=1.)
                self.diffchain.update_T()
                if self.reward_train_idx % 20000 == 0:
                    print("r_d: ", r_d)
                    print("af_p:", self.diffchain.p)
            self.reward_train_idx += 1
        else:
            expert_d = self.discriminator.diffusion.loss(expert_disc_input, disc_ddpm=self.disc_ddpm).unsqueeze(dim=1)
            actor_d = self.discriminator.diffusion.loss(policy_disc_input, disc_ddpm=self.disc_ddpm).unsqueeze(dim=1)
            self.reward_train_idx += 1

        if self.is_clip:
            expert_d = expert_d.clamp(min=1e-8, max=1 - 1e-8)
            actor_d = actor_d.clamp(min=1e-8, max=1 - 1e-8)
        expert_loss = torch.nn.BCELoss()(expert_d, torch.ones(expert_d.size(), device=ptu.device))
        actor_loss = torch.nn.BCELoss()(actor_d, torch.zeros(actor_d.size(), device=ptu.device))

        loss = expert_loss + actor_loss

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)
            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            out = self.discriminator.diffusion.loss(interp_obs, disc_ddpm=self.disc_ddpm).sum()

            gradients = autograd.grad(
                outputs=out,
                inputs=interp_obs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            gradient_penalty = (gradients.norm(2, dim=1) - 1).pow(2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        total_loss = loss + disc_grad_pen_loss

        self.discriminator.diff_opti.zero_grad()
        total_loss.backward()
        if self.is_clip:
            torch.nn.utils.clip_grad_norm_(self.discriminator.model.parameters(), max_norm=self.clip_num)

        self.discriminator.diff_opti.step()
        """
        Save statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval sets this to None so statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc ddpm Loss"] = np.mean(
                ptu.get_numpy(loss)
            )

            self.disc_eval_statistics["Disc ddpm expert Loss"] = np.mean(
                ptu.get_numpy(expert_loss)
            )

            self.disc_eval_statistics["Disc ddpm actor Loss"] = np.mean(
                ptu.get_numpy(actor_loss)
            )
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

            self.disc_eval_statistics["Disc lr"] = self.discriminator.diff_opti.param_groups[0]['lr']

    def _do_reward_training(self, epoch):
        """
        Train the discriminator
        """
        self.disc_optimizer.zero_grad()

        keys = ["observations"]
        if self.state_only:
            keys.append("next_observations")
        else:
            keys.append("actions")
        if self.wrap_absorbing:
            keys.append("absorbing")

        expert_batch = self.get_batch(self.disc_optim_batch_size, 'expert', keys)
        policy_batch = self.get_batch(self.disc_optim_batch_size, 'offpolicy', keys)

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        if self.wrap_absorbing:
            expert_obs = torch.cat(
                [expert_obs, expert_batch["absorbing"][:, 0:1]], dim=-1
            )
            policy_obs = torch.cat(
                [policy_obs, policy_batch["absorbing"][:, 0:1]], dim=-1
            )

        if self.state_only:
            expert_next_obs = expert_batch["next_observations"]
            policy_next_obs = policy_batch["next_observations"]
            if self.wrap_absorbing:
                expert_next_obs = torch.cat(
                    [expert_next_obs, expert_batch["absorbing"][:, 1:]], dim=-1
                )
                policy_next_obs = torch.cat(
                    [policy_next_obs, policy_batch["absorbing"][:, 1:]], dim=-1
                )
            expert_disc_input = torch.cat([expert_obs, expert_next_obs], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_next_obs], dim=1)
        else:
            expert_acts = expert_batch["actions"]
            policy_acts = policy_batch["actions"]
            expert_disc_input = torch.cat([expert_obs, expert_acts], dim=1)
            policy_disc_input = torch.cat([policy_obs, policy_acts], dim=1)
        disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)
        disc_logits = self.discriminator(disc_input)
        disc_preds = (disc_logits > 0).type(disc_logits.data.type())
        disc_ce_loss = self.bce(disc_logits, self.bce_targets)
        accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)
            a = self.discriminator(interp_obs).sum()
            gradients = autograd.grad(
                outputs=self.discriminator(interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        disc_total_loss.backward()
        self.disc_optimizer.step()

        """
        Save statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval sets this to None so statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            self.disc_eval_statistics["Disc Acc"] = np.mean(ptu.get_numpy(accuracy))
            if self.use_grad_pen:
                self.disc_eval_statistics["Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Grad Pen W"] = np.mean(self.grad_pen_weight)

    def _do_policy_training(self, epoch, step_loop):
        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                'offpolicy',
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert, 'expert'
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(self.policy_optim_batch_size, 'offpolicy')

        obs = policy_batch["observations"]
        acts = policy_batch["actions"]
        next_obs = policy_batch["next_observations"]

        if self.wrap_absorbing:
            obs = torch.cat([obs, policy_batch["absorbing"][:, 0:1]], dim=-1)
            next_obs = torch.cat([next_obs, policy_batch["absorbing"][:, 1:]], dim=-1)

        if "ddpm" in self.mode:
            if self.state_only:
                disc_input = torch.cat([obs, next_obs], dim=1)
            else:
                disc_input = torch.cat([acts, obs], dim=1)

            if self._n_train_steps_total >= 3 and self.is_classifier_reward:  # NOTE
                self.discriminator.diffusion.eval()
                if self.clas_type == 'class':
                    disc_logits = self.discriminator.diffusion.reward_classifier(disc_input, step_loop).detach()
                else:
                    disc_logits = self.discriminator.disc_condition_reward(x=disc_input, step_loop=step_loop, change_lambda_factor=self.change_lambda_factor, epoch=epoch).detach()
                self.discriminator.diffusion.train()
            else:
                self.discriminator.diffusion.eval()
                disc_logits = self.discriminator.disc_reward(disc_input).detach()
                self.discriminator.diffusion.train()
        else:
            self.discriminator.eval()
            if self.state_only:
                disc_input = torch.cat([obs, next_obs], dim=1)
            else:
                disc_input = torch.cat([obs, acts], dim=1)

            disc_logits = self.discriminator(disc_input).detach()
            self.discriminator.train()

        # Compute reward
        if self.mode == "gail":
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=-1
            )
        elif self.mode == "ddpm":
            d = disc_logits.to(ptu.device)
            if self.is_reward_smooth:
                raw_rewards = (- torch.log(1 - d + 1e-8)).unsqueeze(dim=1)
                if self.reward_ema is None:
                    self.reward_ema = raw_rewards.mean().detach()
                self.reward_ema = self.reward_smooth_gamma * self.reward_ema + (
                        1 - self.reward_smooth_gamma) * raw_rewards.mean()
                policy_batch["rewards"] = raw_rewards / (self.reward_ema.detach() + 1e-8)
            else:
                policy_batch["rewards"] = (- torch.log(1 - d)).unsqueeze(dim=1)

        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )

        # Policy optimization
        if self.trainer_type == 'REDQ':
            self.policy_trainer.train_step(policy_batch, step_loop, self.stop_UTD)
        elif self.trainer_type == 'sacdelay':
            self.policy_trainer.train_step(policy_batch, step_loop)
        else:
            self.policy_trainer.train_step(policy_batch)

        self.disc_eval_statistics["Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics["Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    @property
    def networks(self):
        return [self.discriminator] + self.policy_trainer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(disc=self.discriminator)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)
