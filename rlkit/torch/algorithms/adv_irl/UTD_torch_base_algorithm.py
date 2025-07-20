import abc
import numpy as np
import torch
from .UTD_base_algorithm import UTDBaseAlgorithm


class UTDTorchBaseAlgorithm(UTDBaseAlgorithm, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def networks(self):
        """
        Used in many settings such as moving to devices
        """
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device):
        for net in self.networks:
            net.to(device)

    @torch.no_grad()
    def evaluate(self, epoch, *args, **kwargs):
        super().evaluate(epoch)
