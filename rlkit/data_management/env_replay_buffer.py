from rlkit.data_management.simple_replay_buffer import (
    SimpleReplayBuffer,
)
from gym.spaces import Box, Discrete, Tuple, Dict
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(self, max_replay_buffer_size, env, random_seed=1995, disc_ddpm=False):
        """
        :param max_replay_buffer_size: 缓冲区最大大小
        :param env: 环境对象
        """
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.ddpm = disc_ddpm
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )

    def add_sample(
            self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        """
        添加单条样本到缓冲区
        """
        super(EnvReplayBuffer, self).add_sample(
            observation, action, reward, terminal, next_observation, **kwargs
        )

    def clear(self):
        """
        清空缓冲区中的所有数据
        """
        self._top = 0
        self._size = 0

    def get_recent(self, batch_size):
        """
        获取最近 batch_size 条数据。
        如果当前缓冲区不足 batch_size 条，则返回所有数据。
        :param batch_size: 要获取的数据数量
        """
        if self._size == 0:
            raise ValueError("Replay buffer is empty.")

        # 计算起始和结束索引
        end_index = self._top
        start_index = (end_index - batch_size) % self._max_replay_buffer_size
        if start_index < 0:
            start_index += self._max_replay_buffer_size

        if self._size < batch_size:
            # 缓冲区数据不足，返回所有数据
            indices = np.arange(0, self._size)
        else:
            # 根据环形缓冲区的索引规则提取数据
            if start_index < end_index:
                indices = np.arange(start_index, end_index)
            else:
                indices = np.concatenate(
                    (np.arange(start_index, self._max_replay_buffer_size), np.arange(0, end_index))
                )

        # 提取数据
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    def replace_with_recent(self, replay_buffer, batch_size):
        """
        从指定的 replay_buffer 中获取最近的 batch_size 条数据，并替换当前缓冲区内容。
        :param replay_buffer: 另一个缓冲区
        :param batch_size: 要获取的数据数量
        """
        # 获取 replay_buffer 中最近的数据
        recent_data = replay_buffer.get_recent(batch_size)

        # 清空当前缓冲区
        self.clear()

        # 添加最近数据到当前缓冲区
        for i in range(len(recent_data["observations"])):
            self.add_sample(
                observation=recent_data["observations"][i],
                action=recent_data["actions"][i],
                reward=recent_data["rewards"][i],
                terminal=recent_data["terminals"][i],
                next_observation=recent_data["next_observations"][i],
            )


def get_dim(space):
    """
    获取空间的维度
    """
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
