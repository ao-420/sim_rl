from collections import deque
import random
import torch
from torch.utils.data import TensorDataset


class ReplayBuffer:
    def __init__(self, max_size, device):
        """
        Replay buffer initialisation

        Parameters:
        - max_size (int): maximum numbers of objects stored by replay buffer
        """
        self.max_size = max_size
        self.device = device
        self.buffer = deque([], maxlen=max_size)

    def get_current_size(self):
        return len(self.buffer)

    def push(self, transition):
        """
        Push a transition to the buffer. If the buffer is full, the oldest transition will be removed.

        Parameters:
        - transition (4-tuple): object to be stored in replay buffer. Should be a tuple of (state, action, reward, next_state),
                                each item in the tuple should be a torch.Tensor.

        Raises:
        - TypeError: If transition is not a tuple.
        - ValueError: If transition is not a 4-tuple.
        """
        if not isinstance(transition, tuple):
            raise TypeError("Transition should be a tuple.")
        if len(transition) != 4:
            raise ValueError("Transition should be a 4-tuple.")
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Get {batch_size} number of random samples from the replay buffer.

        Parameters:
        - batch_size (int): Number of samples to be drawn from the buffer.

        Returns:
        - iterable (list): A list of objects sampled from the buffer without replacement.

        Raises:
        - ValueError: If the buffer is empty, or the sample size is greater than the number of items in the buffer.
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty buffer!")
        if len(self.buffer) < batch_size:
            raise ValueError(
                "Sample size cannot be greater than the number of items contained in the buffer."
            )
        return random.sample(self.buffer, batch_size)

    def get_items(self):
        """
        Returns a 4-tuple containing all items in the buffer (as torch.Tensors).
        """
        state = torch.stack([item[0] for item in self.buffer], dim=0).to(self.device)
        action = torch.stack([item[1] for item in self.buffer], dim=0).to(self.device)
        # reward = torch.stack([item[2].view(-1) for item in self.buffer], dim=0).to(self.device)
        reward = torch.stack(
            [torch.tensor(item[2]).view(-1) for item in self.buffer], dim=0
        ).to(self.device)
        next_state = torch.stack([item[3] for item in self.buffer], dim=0).to(
            self.device
        )
        return TensorDataset(state, action, reward, next_state)

    def clear(self):
        """
        Clear all items from the replay buffer.
        """
        self.buffer = deque([], maxlen=self.max_size)
