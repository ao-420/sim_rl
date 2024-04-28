import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A Residual Block that adds the input (identity) to the output of a linear layer, normalization,
    and activation function.

    Parameters:
    - in_features (int): Number of input features.
    - out_features (int): Number of output features.

    Attributes:
    - linear (nn.Linear): Linear transformation layer.
    - norm (nn.LayerNorm): Layer normalization.
    - activation (nn.LeakyReLU): Leaky ReLU activation function.
    """

    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass for the Residual Block.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying the block's layers and adding the input tensor.
        """
        identity = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        if x.size(-1) != out.size(-1):
            linear_layer = nn.Linear(x.size(-1), out.size(-1)).to(x.device)
            identity = linear_layer(identity)
        out += identity  # Skip connection
        return out


class Actor(nn.Module):
    """
    An Actor network for actor-critic algorithms, using a sequence of layers and optional residual blocks.

    Parameters:
    - n_states (int): Number of states in the input space.
    - n_actions (int): Number of actions in the output space.
    - hidden (list): List of integers defining the number of nodes in each hidden layer.
    - device (torch.device): The device tensors will be sent to for calculations.

    Attributes:
    - layers (nn.Sequential): Sequential container of layers.
    - device (torch.device): Device on which the network will run.
    """

    def __init__(self, n_states, n_actions, hidden, device):
        super(Actor, self).__init__()
        check_validity(hidden)

        layers = [
            nn.Linear(n_states, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.LeakyReLU(0.2),
        ]

        for i in range(1, len(hidden)):
            layers.append(ResidualBlock(hidden[i - 1], hidden[i]))

        layers.append(nn.Linear(hidden[-1], n_actions))
        layers.append(nn.Sigmoid())  # Ensure output is between 0 and 1
        self.layers = nn.Sequential(*layers)
        self.device = device

    def forward(self, state):
        """
        Forward pass for the Actor network.

        Parameters:
        - state (torch.Tensor or array-like): Input state.

        Returns:
        - torch.Tensor: Action values as output from the network.
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().detach().to(self.device).float()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = self.layers(state_tensor)

        return action


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden, device):
        """
        Neural network representing the critic (denoted Q in literature). Given a
        state vector and action vector as input, return the Q value of this
        state-action pair.

        Parameters:
        - n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
        - hidden (list of ints):  Number of neurons in each hidden layer. len(hidden) = number of
                                    hidden layers within the network.
        """
        super(Critic, self).__init__()
        check_validity(hidden)

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden[0] + n_actions, hidden[1]), nn.LeakyReLU(0.2)
        )
        layers = []

        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i - 1], hidden[i]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden[-1], 1))

        self.layer3 = nn.Sequential(*layers)
        self.device = device

    def forward(self, xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have shape
                        (N,1)

        Returns:
        - torch.Tensor: Output Q value.
        """
        x, a = xa
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if type(a) == np.ndarray:
            a = torch.tensor(x)

        x = x.float()
        out = self.layer1(x.to(self.device))

        if len(a.shape) == 1:
            out = self.layer2(torch.cat([out, a]))
        else:
            out = self.layer2(torch.cat([out, a], 1))
        out = self.layer3(out)
        return out


class RewardModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden, device):
        """
        Neural network representing the DDPG agent's internal model of the environment.
        Given a state vector and action vector as input, returns the predicted reward
        of this state-action pair.

        Parameters:
        - n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
        - n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
        - hidden (list of ints):    Number of neurons in each hidden layer. len(hidden) = number of
                                    hidden layers within the network.
        """
        super().__init__()
        check_validity(hidden)
        self.device = device

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden[0] + n_actions, hidden[1]), nn.LeakyReLU(0.2)
        )
        layers = []

        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i - 1], hidden[i]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden[-1], 1))  # scalar output
        layers.append(nn.LeakyReLU())
        self.layer3 = nn.Sequential(*layers)

    def forward(self, xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have
                        shape (N,1)

        Returns:
        - torch.Tensor : predicted reward of state-action pair
        """
        x, a = xa
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if type(a) == np.ndarray:
            a = torch.tensor(x)
        x = x.float()

        out = self.layer1(x.to(self.device))
        if len(a.shape) == 1:
            out = self.layer2(torch.cat([out, a]).to(self.device))
        else:
            out = self.layer2(torch.cat([out, a], 1).to(self.device))
        out = self.layer3(out)
        return out


class NextStateModel(nn.Module):
    def __init__(self, n_states, n_actions, hidden, device):
        """
        .       Neural network representing the DDPG agent's internal model of the environment.
                Given a state vector and action vector as input, returns the predicted next
                state of this state-action pair.

                Parameters:
                - n_states (int)  : Number of nodes in the system. Represents the size of the state vector.
                - n_actions (int) : Number of nodes in the system. Represents the size of the action vector.
                - hidden (list of ints):    Number of neurons in each hidden layer. len(hidden) = number of
                                            hidden layers within the network.
        """
        super().__init__()
        self.device = device
        check_validity(hidden)

        self.layer1 = nn.Sequential(nn.Linear(n_states, hidden[0]), nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden[0] + n_actions, hidden[1]), nn.LeakyReLU(0.2)
        )
        layers = []

        for i in range(2, len(hidden)):
            layers.append(nn.Linear(hidden[i - 1], hidden[i]))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden[-1], n_states))  # vector output
        layers.append(nn.LeakyReLU(0.2))
        self.layer3 = nn.Sequential(*layers)

    def forward(self, xa):
        """
        Parameters:
        - xa (list):    [state vector, action vector] where both vectors have
                        shape (N,1)

        Returns:
        - torch.Tensor (n_states,) : predicted next state
        """
        x, a = xa
        if type(x) == np.ndarray:
            x = torch.tensor(x.to(self.device))
        if type(a) == np.ndarray:
            a = torch.tensor(x.to(self.device))
        x = x.float()
        out = self.layer1(x)

        if len(a.shape) == 1:
            out = self.layer2(torch.cat([out, a]))
        else:
            out = self.layer2(torch.cat([out, a], 1))
        out = self.layer3(out)
        return out


def check_validity(hidden):
    """
    Helper function that checks the validity of the input 'hidden' to the
    constructors of the neural networks
    """
    if type(hidden) != list or not all(isinstance(x, int) for x in hidden):
        raise Exception("The argument 'hidden' should be a list of integers.")
    if len(hidden) < 2:
        raise Exception("The list/tuple should have a length >= 2")
