from collections import OrderedDict
import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim
from torch import Tensor
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from .core import TanhGaussianPolicy, Mlp, soft_update_model1_with_model2, ReplayBuffer, mbpo_target_entropy_dict


class REDQSoftActorCritic:
    def __init__(
            self,
            policy: nn.Module,
            obs_dim,
            act_dim,
            hidden_sizes=(256, 256),
            lr=3e-4,
            gamma=0.99,
            polyak=0.995,
            alpha=0.2,
            auto_alpha=True,
            target_entropy=None,
            num_Q=10,
            num_min=2,
            target_drop_rate=0.0,
            layer_norm=False,
            utd_ratio=20,
            policy_update_delay=20,
            num_update=20000,
            **kwargs
    ):
        self.policy = policy
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_Q = num_Q
        self.num_min = num_min
        self.gamma = gamma
        self.polyak = polyak
        self.utd_ratio = utd_ratio
        self.policy_update_delay = policy_update_delay
        self.target_drop_rate = target_drop_rate
        self.layer_norm = layer_norm
        self.num_update = num_update


        # Initialize multiple Q networks and targets
        self.q_nets = [Mlp(obs_dim + act_dim, 1, hidden_sizes,
                           target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(self.device) for _ in
                       range(num_Q)]
        self.q_target_nets = [Mlp(obs_dim + act_dim, 1, hidden_sizes,
                                  target_drop_rate=target_drop_rate, layer_norm=layer_norm).to(self.device) for _ in
                              range(num_Q)]
        for q_net, q_target in zip(self.q_nets, self.q_target_nets):
            q_target.load_state_dict(q_net.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizers = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_nets]

        # Alpha (entropy temperature) optimization
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            if target_entropy is None:
                self.target_entropy = -np.prod(kwargs["env"].action_space.shape) / 2.0
            else:
                self.target_entropy = target_entropy
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
        # Store evaluation statistics
        self.eval_statistics = None

    def train_step(self, batch, step_loop, stop_UTD):
        obs, actions, rewards, next_obs, dones = (batch["observations"], batch["actions"],
                                                  batch["rewards"], batch["next_observations"], batch["terminals"])
        if stop_UTD:
            self.utd_ratio = 1
            self.policy_update_delay = 1
        # Compute REDQ target
        with torch.no_grad():
            # Get next actions and log probabilities from policy (similar to SAC)
            next_policy_outputs = self.policy(next_obs, return_log_prob=True)
            next_new_actions, next_policy_mean, next_policy_log_std, next_log_pi = next_policy_outputs[:4]

            # Sample Q values from target networks
            sampled_indices = np.random.choice(self.num_Q, self.num_min, replace=False)
            min_q_vals = torch.min(torch.stack([
                self.q_target_nets[i](torch.cat([next_obs, next_new_actions], dim=-1))
                for i in sampled_indices], dim=0), dim=0).values
            q_target = rewards + self.gamma * (1.0 - dones) * (min_q_vals - self.alpha * next_log_pi)

        # Update Q networks
        q_losses = []
        for i, (q_net, q_optimizer) in enumerate(zip(self.q_nets, self.q_optimizers)):
            q_values = q_net(torch.cat([obs, actions], dim=-1))
            q_loss = nn.MSELoss()(q_values, q_target)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()
            q_losses.append(q_loss.item())

        # Update policy and alpha
        if ((step_loop + 1) % self.policy_update_delay == 0) or step_loop == self.num_update - 1:
            # Sample actions and log probabilities from policy
            # 获取策略输出
            next_policy_outputs = self.policy(obs, return_log_prob=True)
            actions_sampled, policy_mean, policy_log_std, log_prob_sampled = next_policy_outputs[:4]
            # 计算平均Q值
            avg_q = torch.mean(
                torch.stack([q_net(torch.cat([obs, actions_sampled], dim=-1)) for q_net in self.q_nets], dim=0), dim=0)

            # 计算策略损失
            policy_loss = (self.alpha * log_prob_sampled - avg_q).mean()

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()



            if self.auto_alpha:
                # Update alpha based on target entropy
                alpha_loss = -(self.log_alpha * (log_prob_sampled + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        # Soft update of target Q networks
        for q_net, q_target_net in zip(self.q_nets, self.q_target_nets):
            soft_update_model1_with_model2(q_target_net, q_net, self.polyak)


        # Save evaluation statistics
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        self.eval_statistics["Reward Scale"] = self.reward_scale if hasattr(self, 'reward_scale') else None
        self.eval_statistics["Q Losses"] = np.mean(q_losses)  # Ensure q_losses is a list or tensor


        # Update with Q network predictions, alpha, and log probabilities


        # Collect gradients for parameters if available
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                self.eval_statistics["policy " + name] = ptu.get_numpy(grad_norm)

        for name, param in self.q_nets[0].named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                self.eval_statistics["q1 " + name] = ptu.get_numpy(grad_norm)

        for name, param in self.q_target_nets[0].named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm()
                self.eval_statistics["target_q1 " + name] = ptu.get_numpy(grad_norm)




    @property
    def networks(self):
        """
        Return the list of networks used in training.
        In REDQ, we have multiple Q networks, their corresponding target networks, and the policy network.
        """
        return [
            self.policy,
            *self.q_nets,
            *self.q_target_nets,
        ]

    def _update_target_network(self):
        """
        Perform soft updates of the Q target networks from the Q networks.
        REDQ uses multiple Q networks, so we update all of them.
        """
        for q_net, q_target_net in zip(self.q_nets, self.q_target_nets):
            ptu.soft_update_from_to(q_net, q_target_net, self.polyak)

    def get_snapshot(self):
        """
        Save the state of the current model, including Q networks, target Q networks, policy, and optimizers.
        """
        snapshot = dict(
            policy=self.policy,
            log_alpha=self.log_alpha,
            policy_optimizer=self.policy_optimizer,
            alpha_optimizer=self.alpha_optimizer,
        )

        # Save each Q network and its target network
        for i, (q_net, q_target_net) in enumerate(zip(self.q_nets, self.q_target_nets)):
            snapshot[f"qf{i + 1}"] = q_net
            snapshot[f"target_qf{i + 1}"] = q_target_net

        # If using automatic alpha, save it
        if self.auto_alpha:
            snapshot["log_alpha"] = self.log_alpha
            snapshot["alpha_optimizer"] = self.alpha_optimizer

        return snapshot

    def load_snapshot(self, snapshot):
        """
        Load the state of the model from a snapshot. This includes the Q networks, target Q networks, and policy.
        """
        self.policy = snapshot["policy"]
        self.log_alpha = snapshot["log_alpha"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.alpha_optimizer = snapshot["alpha_optimizer"]

        # Load each Q network and its target network
        for i in range(self.num_Q):
            self.q_nets[i] = snapshot[f"qf{i + 1}"]
            self.q_target_nets[i] = snapshot[f"target_qf{i + 1}"]

        if self.auto_alpha:
            self.log_alpha = snapshot["log_alpha"]
            self.alpha_optimizer = snapshot["alpha_optimizer"]

    def get_eval_statistics(self):
        """
        Return the evaluation statistics for the current training step.
        These statistics are computed during training.
        """
        if self.eval_statistics is None:
            return {}

        return self.eval_statistics

    def end_epoch(self):
        """
        Reset evaluation statistics at the end of an epoch.
        This allows us to recompute statistics on the next epoch.
        """
        self.eval_statistics = None

    # def to(self, device):
    #     """
    #     Move all model components (policy, Q networks, target Q networks, and other tensors) to the specified device.
    #     This is important for training on GPU.
    #     """
    #     self.device = device
    #     self.log_alpha.to(device)
    #
    #     # Move each Q network and target Q network to the device
    #     for q_net, q_target_net in zip(self.q_nets, self.q_target_nets):
    #         q_net.to(device)
    #         q_target_net.to(device)
    #
    #     # Move the policy network to the device
    #     self.policy.to(device)
    #
    #     # Move the optimizers and any other necessary tensors
    #     self.policy_optimizer.param_groups[0]['params'] = [param.to(device) for param in self.policy.parameters()]
    #     for q_optimizer in self.q_optimizers:
    #         for param_group in q_optimizer.param_groups:
    #             param_group['params'] = [param.to(device) for param in param_group['params']]
    #
    #     # Move alpha optimizer if it exists
    #     if self.alpha_optimizer is not None:
    #         for param_group in self.alpha_optimizer.param_groups:
    #             param_group['params'] = [param.to(device) for param in param_group['params']]

    def to(self, device):
        self.log_alpha.to(device)
        for net in self.networks:
            net.to(device)

    def get_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _, _, _, _, _ = self.policy(obs, deterministic=deterministic)
        return action.cpu().detach().numpy()[0]
