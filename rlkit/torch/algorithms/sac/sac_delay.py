from collections import OrderedDict

import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic_dalay(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions
        - has auto-tuned alpha
    """

    def __init__(
            self,
            policy: nn.Module,
            qf1: nn.Module,
            qf2: nn.Module,
            reward_scale=1.0,
            discount=0.99,
            policy_lr=1e-3,
            qf_lr=1e-3,
            alpha_lr=3e-4,
            soft_target_tau=1e-2,
            alpha=0.2,
            train_alpha=True,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            optimizer_class=optim.Adam,
            beta_1=0.9,
            target_entropy=None,
            policy_update_delay=20,
            num_update=1000,
            **kwargs
    ):
        self.num_update = num_update
        self.policy_update_delay = policy_update_delay

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.train_alpha = train_alpha
        self.log_alpha = torch.tensor(
            np.log(alpha), requires_grad=train_alpha, device=ptu.device
        )
        self.alpha = self.log_alpha.detach().exp()
        assert "env" in kwargs.keys(), "env info should be taken into SAC alpha"
        self.target_entropy = target_entropy
        if target_entropy is None:
            self.target_entropy = -np.prod(kwargs["env"].action_space.shape) / 2.0

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha], lr=alpha_lr, betas=(beta_1, 0.999)
        )

    def train_step(self, batch, step_loop):
        # q_params = itertools.chain(self.qf1.parameters(), self.qf2.parameters())
        # policy_params = itertools.chain(self.policy.parameters())

        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        # Make sure policy accounts for squashing functions like tanh correctly!
        next_policy_outputs = self.policy(next_obs, return_log_prob=True)
        # in this part, we only need new_actions and log_pi with no grad
        (
            next_new_actions,
            next_policy_mean,
            next_policy_log_std,
            next_log_pi,
        ) = next_policy_outputs[:4]
        target_qf1_values = self.target_qf1(
            next_obs, next_new_actions
        )  # do not need grad || it's the shared part of two calculation
        target_qf2_values = self.target_qf2(
            next_obs, next_new_actions
        )  # do not need grad || it's the shared part of two calculation
        min_target_value = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * (
                min_target_value - self.alpha * next_log_pi
        )  ## original implementation has detach
        q_target = q_target.detach()

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        # freeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = False

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        if ((step_loop + 1) % self.policy_update_delay == 0) or step_loop == self.num_update - 1:
            """
                    Policy Loss
            """
            # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
            #     p.requires_grad = False
            # for p in self.policy.parameters():
            #     p.requires_grad = True
            policy_outputs = self.policy(obs, return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            q1_new_acts = self.qf1(obs, new_actions)
            q2_new_acts = self.qf2(obs, new_actions)  ## error
            q_new_actions = torch.min(q1_new_acts, q2_new_acts)

            self.policy_optimizer.zero_grad()
            policy_loss = torch.mean(self.alpha * log_pi - q_new_actions)  ##
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
            policy_loss = policy_loss + policy_reg_loss
            policy_loss.backward()
            self.policy_optimizer.step()

            """
            Update alpha
            """
            if self.train_alpha:
                log_prob = log_pi.detach() + self.target_entropy
                alpha_loss = -(self.log_alpha * log_prob).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.detach().exp()

        """
        Update networks
        """
        # unfreeze all -> initial states
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = True

        # unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True

        self._update_target_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Reward Scale"] = self.reward_scale
            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            # if self.train_alpha:
            #     self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
            # self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q1 Predictions",
                    ptu.get_numpy(q1_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Q2 Predictions",
                    ptu.get_numpy(q2_pred),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Alpha",
                    [ptu.get_numpy(self.alpha)],
                )
            )
            # self.eval_statistics.update(
            #     create_stats_ordered_dict(
            #         "Log Pis",
            #         ptu.get_numpy(log_pi),
            #     )
            # )
            # self.eval_statistics.update(
            #     create_stats_ordered_dict(
            #         "Policy mu",
            #         ptu.get_numpy(policy_mean),
            #     )
            # )
            # self.eval_statistics.update(
            #     create_stats_ordered_dict(
            #         "Policy log std",
            #         ptu.get_numpy(policy_log_std),
            #     )
            # )

            for name, param in self.qf1.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics["qf1 " + name] = ptu.get_numpy(grad_norm)

            for name, param in self.qf2.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics["qf2 " + name] = ptu.get_numpy(grad_norm)

            for name, param in self.target_qf1.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics["target_qf1 " + name] = ptu.get_numpy(grad_norm)

            for name, param in self.target_qf2.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics["target_qf2 " + name] = ptu.get_numpy(grad_norm)

            for name, param in self.policy.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm()
                    self.eval_statistics["policy " + name] = ptu.get_numpy(grad_norm)

            # for name, param in self.log_alpha.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm()
            #         self.eval_statistics["log_alpha " + name] = ptu.get_numpy(grad_norm)

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            log_alpha=self.log_alpha,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
            alpha_optimizer=self.alpha_optimizer,
        )

    def load_snapshot(self, snapshot):
        self.qf1 = snapshot["qf1"]
        self.qf2 = snapshot["qf2"]
        self.policy = snapshot["policy"]
        self.target_qf1 = snapshot["target_qf1"]
        self.target_qf2 = snapshot["target_qf2"]
        self.log_alpha = snapshot["log_alpha"]
        self.policy_optimizer = snapshot["policy_optimizer"]
        self.qf1_optimizer = snapshot["qf1_optimizer"]
        self.qf2_optimizer = snapshot["qf2_optimizer"]
        self.alpha_optimizer = snapshot["alpha_optimizer"]

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None

    def to(self, device):
        self.log_alpha.to(device)
        for net in self.networks:
            net.to(device)
