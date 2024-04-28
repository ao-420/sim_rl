import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from unittest.mock import patch, MagicMock

from evaluation.convergence_evaluation.convergence_evaluation import ConvergenceEvaluation


@pytest.fixture
def confidence_instance():
    # Assume window_size, threshold, consecutive_points, timesteps, num_sim are parameters of ConvergenceEvaluation
    return ConvergenceEvaluation(window_size=5, threshold=0.1, consecutive_points=3, timesteps=5, num_sim=5)

@patch("foundations.core_functions.load_hyperparams")
@patch("foundations.core_functions.create_ddpg_agent")
@patch("foundations.core_functions.create_simulation_env")
@patch("builtins.open", new_callable=MagicMock)
def test_start_train(
    mock_open,
    mock_create_simulation_env,
    mock_create_ddpg_agent,
    mock_load_hyperparams,
    confidence_instance
):
    # Mock returns
    mock_env = MagicMock()
    mock_agent = MagicMock()
    mock_create_simulation_env.return_value = mock_env
    mock_create_ddpg_agent.return_value = mock_agent
    mock_load_hyperparams.return_value = ({}, {})  # Returning empty dictionaries for hyperparameters

    # Mock file open to prevent FileNotFoundError
    mock_open.return_value.__enter__.return_value = MagicMock()

    # Assume confidence_instance should simulate several episodes
    confidence_instance.num_episodes = [1] * 5  # Define the number of episodes for testing

    # Mock methods within ConvergenceEvaluation that are not under direct test to isolate behavior
    with patch.object(confidence_instance, 'train', return_value=(None, [], [], {}, {}, {}, None)) as mock_train, \
         patch.object(confidence_instance, 'evaluate_convergence', return_value=100) as mock_evaluate:

        # Execute the method under test
        confidence_instance.start_train(
            "fake_config.yml", "fake_eval_config.yml", "fake_param_file.yml"
        )

        # Assertions
        assert mock_train.call_count == len(confidence_instance.num_episodes)
        assert mock_evaluate.call_count == len(confidence_instance.num_episodes)


def test_evaluate_agent_no_stabilization(confidence_instance):
    # Mock the environment and agent needed for evaluation
    mock_env = MagicMock()
    mock_env.get_state.return_value = [0]  # Simplistic state
    mock_env.get_next_state.return_value = ([0],)  # Simplistic next state
    mock_env.get_reward.return_value = 1  # Constant reward
    mock_agent = MagicMock()
    mock_agent.actor.return_value = MagicMock(
        detach=MagicMock(return_value=[0])
    )  # Simplistic action

    # Simulate reward data that should not meet the stabilization criteria
    reward_data = {'episodes': list(range(100)), 'rewards': [i % 10 for i in range(100)]}

    # Inject this reward data into the instance
    confidence_instance.reward_data_from_json = reward_data

    stabilization_point = confidence_instance.evaluate_convergence()
    expected_stabilization_point = 1  # Expect no stabilization due to fluctuating rewards

    assert stabilization_point == expected_stabilization_point, f"Stabilization point did not match expected value. Found: {stabilization_point}"

@patch("matplotlib.pyplot.savefig")
def test_save_reward_plot(mock_savefig, confidence_instance):
    # Assume the total_rewards have been populated after some hypothetical runs
    confidence_instance.total_rewards = [100, 200, 300]
    confidence_instance.num_episodes = [10, 50, 100]
    confidence_instance.plot_results(confidence_instance.total_rewards, confidence_instance.num_episodes)
    mock_savefig.assert_called_once()  # Check if the plot is actually saved

# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_start_train"])
    