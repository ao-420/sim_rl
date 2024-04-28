import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch
from agents.model import *
import pytest
from unittest.mock import MagicMock, patch

device = torch.device("cpu")


@pytest.mark.parametrize(
    "n_states,n_actions,hidden,device",
    [
        (10, 2, [64, 64], device),
        (20, 4, [128, 128], device),
    ],
)
def test_actor_output_shape(n_states, n_actions, hidden, device):
    """
    Test the Actor model to ensure it produces the correct output shape and values range.

    Parameters:
    - n_states (int): Number of states in the environment.
    - n_actions (int): Number of possible actions.
    - hidden (list): List of integers representing the size of hidden layers.
    """
    model = Actor(n_states, n_actions, hidden, device)
    sample_input = torch.rand(size=(1, n_states), device=device)
    output = model(sample_input)
    assert output.shape == (1, n_actions), "Actor output shape is incorrect"

    # Additionally, check if outputs are within the expected range [0, 1]
    assert torch.all(0 <= output) and torch.all(
        output <= 1
    ), "Actor output values not in [0, 1]"


@pytest.mark.parametrize(
    "n_states,n_actions,hidden,device",
    [
        (10, 2, [64, 64], device),
        (20, 4, [128, 128], device),
    ],
)
def test_critic_output_shape(n_states, n_actions, hidden, device):
    """
    Test the Critic model for correct output shape based on given inputs.

    Parameters:
    - n_states (int): Number of states in the environment.
    - n_actions (int): Number of possible actions.
    - hidden (list): List of integers representing the size of hidden layers.
    """
    model = Critic(n_states, n_actions, hidden, device)
    sample_state = torch.rand(size=(1, n_states), device=device)
    sample_action = torch.rand(size=(1, n_actions), device = device)
    output = model([sample_state, sample_action])
    assert output.shape == (1, 1), "Critic output shape is incorrect"


@pytest.mark.parametrize(
    "n_states,n_actions,hidden,device",
    [
        (10, 2, [64, 64], device),
        (20, 4, [128, 128], device),
    ],
)
def test_reward_model_output_shape(n_states, n_actions, hidden, device):
    """
    Validate the output shape of the RewardModel given the environment state and action.

    Parameters:
    - n_states (int): Number of states in the environment.
    - n_actions (int): Number of possible actions.
    - hidden (list): List of integers representing the size of hidden layers.
    """
    model = RewardModel(n_states, n_actions, hidden, device)
    sample_state = torch.rand(size=(1, n_states))
    sample_action = torch.rand(size=(1, n_actions))
    output = model([sample_state, sample_action])
    assert output.shape == (1, 1), "RewardModel output shape is incorrect"


@pytest.mark.parametrize(
    "n_states,n_actions,hidden,device",
    [
        (10, 2, [64, 64], device),
        (20, 4, [128, 128], device),
    ],
)
def test_next_state_model_output_shape(n_states, n_actions, hidden, device):
    """
    Test the NextStateModel for generating correct output shape, simulating the next state.

    Parameters:
    - n_states (int): Number of states in the environment.
    - n_actions (int): Number of possible actions.
    - hidden (list): List of integers representing the size of hidden layers.
    """
    model = NextStateModel(n_states, n_actions, hidden, device)
    sample_state = torch.rand(size=(1, n_states))
    sample_action = torch.rand(size=(1, n_actions))
    output = model([sample_state, sample_action])
    assert output.shape == (1, n_states), "NextStateModel output shape is incorrect"


def check_validity(hidden):
    """
    Helper function that checks the validity of the input 'hidden' to the
    constructors of the neural networks.

    Parameters:
        hidden (list): A list of integers specifying the number of neurons in each hidden layer.

    Raises:
        ValueError: If 'hidden' is not a list of integers or its length is less than 2.
    """
    if not isinstance(hidden, list) or not all(isinstance(x, int) for x in hidden):
        raise ValueError("The argument 'hidden' should be a list of integers.")
    if len(hidden) < 2:
        raise ValueError("The list should have a length >= 2.")


def test_check_validity():
    """
    Validate the 'check_validity' helper function that checks the integrity of the 'hidden' parameter.

    This function ensures that the 'hidden' parameter passed to model constructors is a list of integers
    of length >= 2, throwing appropriate exceptions otherwise.
    """
    try:
        check_validity([64, 64])
    except Exception as e:
        pytest.fail(f"Unexpected exception for valid input: {e}")

    # Invalid input: not a list
    with pytest.raises(Exception):
        check_validity("not a list")

    # Invalid input: contains non-integers
    with pytest.raises(Exception):
        check_validity([64, "not an integer"])

    # Invalid input: length < 2
    with pytest.raises(Exception):
        check_validity([64])


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
