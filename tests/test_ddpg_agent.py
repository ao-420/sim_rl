import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

from collections import OrderedDict
from unittest.mock import MagicMock, patch
import pytest
from agents.model import Actor, Critic, RewardModel, NextStateModel
from agents.ddpg_agent import DDPGAgent
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_states = 10
n_actions = 2
hidden = {
    "actor": [64, 64],
    "critic": [64, 64],
    "reward_model": [10, 10],
    "next_state_model": [10, 10],
}
params = {
    "tau": 0.1,
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "batch_size": 20,
    "discount": 0.99,
    "epsilon": 0.1,
    "planning_steps": 5,
    "planning_std": 0.1,
    "buffer_size": 5000,
}


@pytest.fixture
def ddpg_agent():
    """
    Pytest fixture to setup DDPGAgent for testing.
    """
    return DDPGAgent(n_states, n_actions, hidden, params, device)


# seed setting for reproducibility
def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# set constants for weight and bias initialization for testing
def init_weights_constant(m, WEIGHT_VALUE=0.5, BIAS_VALUE=0.1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, WEIGHT_VALUE)
        nn.init.constant_(m.bias, BIAS_VALUE)


# compare state dictionaries
def compare_state_dicts(state_dict1, state_dict2):
    assert isinstance(state_dict1, OrderedDict), "state_dict1 is not an OrderedDict."
    assert isinstance(state_dict2, OrderedDict), "state_dict2 is not an OrderedDict."
    for (k1, v1), (k2, v2) in zip(state_dict1.items(), state_dict2.items()):
        assert k1 == k2, "The keys of the state_dict are different."
        # print("-"*50)
        # print(k1)
        # print(v1)
        # print("\n\n", k2)
        # print(v2)
        assert torch.equal(v1, v2), "The values of the state_dict are different."


def test_fit_model(ddpg_agent):
    """
    Test the `fit_model` method of the DDPGAgent to ensure it processes the training data correctly,
    applying model updates as expected. This test uses a mock of the Adam optimizer's step method
    to prevent actual parameter updates during the test.
    """
    with patch("torch.optim.Adam.step") as mock_step:
        mock_step.return_value = None

        for _ in range(10):
            state = torch.randn(n_states)
            action = torch.randn(n_actions)
            reward = torch.randn(1)
            next_state = torch.randn(n_states)
            ddpg_agent.buffer.push((state, action, reward, next_state))

        initial_loss_reward, initial_loss_next_state = ddpg_agent.fit_model(
            batch_size=5, epochs=1
        )
        assert (
            len(initial_loss_reward) > 0
        ), "No reward model training loss was recorded."
        assert (
            len(initial_loss_next_state) > 0
        ), "No next state model training loss was recorded."


def test_select_action(ddpg_agent):
    """
    Test the `select_action` method of the DDPGAgent to ensure it returns an action of the correct shape
    given a state input. This test verifies the action selection process is functioning as expected.
    """
    state = torch.randn(n_states)
    action = ddpg_agent.select_action(state)
    assert action.shape == torch.Size([2]), "Action shape mismatch."


def test_update_q_values(ddpg_agent):
    """
    Test the Q-value updating process within the critic network of the DDPGAgent. This method tests
    the critic's ability to update its Q-values by observing changes in loss before and after an update cycle,
    ensuring the network learns from the provided batch of experiences.
    """
    set_torch_seed(42)
    batch = [
        (
            torch.randn(n_states),
            torch.randn(n_actions),
            torch.randn(1),
            torch.randn(n_states),
        )
        for _ in range(5)
    ]
    initial_loss = ddpg_agent.update_critic_network(batch)

    updated_loss = ddpg_agent.update_critic_network(batch)
    assert (
        updated_loss < initial_loss
    ), "Critic network Q-value update did not reduce loss."


def test_update_actor_network(ddpg_agent, num_experiences=100):
    """
    Tests the update_actor_network method in the DDPGAgent class. Steps to achieve this:
        1. Setting the torch seed for reproducibility and then creating fake data.
        2. Setting weights for Actor and Critic in DDPGAgent class (both policy and target networks).
        3. Create the necessary baseline models for comparison (here: actor and critic policy networks)
        4. Call the update_actor_network method to update the actor network in DDPGAgent.
        5. Update the actor network in the baseline model manually
        6. Compare the state dictionaries of the two actor networks (baseline and DDPGAgent) before and after updates.
    """
    # set torch seed and create fake data
    set_torch_seed(42)
    batch = [
        (
            torch.randn(n_states),
            torch.randn(n_actions),
            torch.randn(1),
            torch.randn(n_states),
        )
        for _ in range(num_experiences)
    ]

    # set weights for Actor and Critic in DDPGAgent class
    ddpg_agent.actor.apply(init_weights_constant)
    ddpg_agent.hard_update(network="actor")
    ddpg_agent.critic.apply(init_weights_constant)
    ddpg_agent.hard_update(network="critic")

    # create baseline model for comparison (Actor)
    baseline_actor = Actor(n_states, n_actions, hidden["actor"], device)
    baseline_optim = torch.optim.Adam(
        baseline_actor.parameters(), lr=params["actor_lr"]
    )
    baseline_actor.apply(init_weights_constant)
    baseline_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        baseline_optim, gamma=0.8
    )

    # create baseline model for comparison (Critic)
    baseline_critic = Critic(n_states, n_actions, hidden["critic"], device)
    baseline_critic.apply(init_weights_constant)

    # compare state dictionaries before updates
    compare_state_dicts(baseline_actor.state_dict(), ddpg_agent.actor.state_dict())
    compare_state_dicts(baseline_critic.state_dict(), ddpg_agent.critic.state_dict())

    # update actor network in DDPGAgent
    ddpg_agent.update_actor_network(batch)

    # update actor network in baseline model
    total_policy_loss = torch.zeros(1, requires_grad=True)
    baseline_optim.zero_grad()
    for experience in batch:
        state, action, reward, next_state = experience
        # self.actor.zero_grad()
        policy_loss = -baseline_critic([state, baseline_actor(state)])
        policy_loss = policy_loss.mean().to(torch.float32)
        total_policy_loss = total_policy_loss + policy_loss
    mean_policy_loss = total_policy_loss / len(batch)
    mean_policy_loss.backward()
    baseline_optim.step()
    baseline_scheduler.step()

    # compare state dictionaries after updates
    compare_state_dicts(baseline_actor.state_dict(), ddpg_agent.actor.state_dict())
    compare_state_dicts(baseline_critic.state_dict(), ddpg_agent.critic.state_dict())


def test_update_critic_network(ddpg_agent, num_experiences=100):
    """
    Tests the update_critic_network method in the DDPGAgent class. Steps to achieve this:
        1. Setting the torch seed for reproducibility and then creating fake data.
        2. Setting weights for Actor and Critic in DDPGAgent class (both policy and target networks).
        3. Create the necessary baseline models for comparison (here: actor target network, critic policy and target networks)
        4. Call the update_critic_network method to update the critic network in DDPGAgent.
        5. Update the critic network in the baseline model manually
        6. Compare the state dictionaries of the two critic networks (baseline and DDPGAgent) before and after updates.
    """

    # set torch seed and create fake data
    set_torch_seed(42)
    batch = [
        (
            torch.randn(n_states),
            torch.randn(n_actions),
            torch.randn(1),
            torch.randn(n_states),
        )
        for _ in range(num_experiences)
    ]

    # set weights for Actor and Critic in DDPGAgent class
    ddpg_agent.actor.apply(init_weights_constant)
    ddpg_agent.hard_update(network="actor")
    ddpg_agent.critic.apply(init_weights_constant)
    ddpg_agent.hard_update(network="critic")

    # create baseline model for comparison (Actor)
    baseline_actor_target = Actor(n_states, n_actions, hidden["actor"], device)
    baseline_actor_target.apply(init_weights_constant)

    # create baseline model for comparison (Critic)
    baseline_critic = Critic(n_states, n_actions, hidden["critic"], device)
    baseline_critic_target = Critic(n_states, n_actions, hidden["critic"], device)
    baseline_optim = torch.optim.Adam(
        baseline_critic.parameters(), lr=params["critic_lr"]
    )
    baseline_critic.apply(init_weights_constant)
    baseline_critic_target.apply(init_weights_constant)

    # compare state dictionaries before updates
    compare_state_dicts(baseline_critic.state_dict(), ddpg_agent.critic.state_dict())

    # update critic network in DDPGAgent
    ddpg_agent.update_critic_network(batch)

    # update critic network in baseline model
    total_critic_loss = torch.zeros(1, requires_grad=True)
    baseline_optim.zero_grad()
    for experience in batch:
        state, action, reward, next_state = experience
        next_q_value = baseline_critic_target(
            [next_state, baseline_actor_target(next_state).detach()]
        ).detach()
        target_q = reward + params["discount"] * next_q_value
        actual_q = baseline_critic([state, action])
        q_loss = nn.MSELoss()(actual_q.to(torch.float32), target_q.to(torch.float32))
        # q_loss.backward(retain_graph=True) # needed cause called multiple times in the same update in plan
        total_critic_loss = total_critic_loss + q_loss
    mean_critic_loss = total_critic_loss / len(batch)
    mean_critic_loss.backward()
    baseline_optim.step()

    # compare state dictionaries after updates
    compare_state_dicts(baseline_critic.state_dict(), ddpg_agent.critic.state_dict())


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()