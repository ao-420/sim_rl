import sys
from pathlib import Path
import concurrent.futures

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import torch
from foundations.core_functions import *
import pytest
from unittest.mock import MagicMock, patch
import yaml
import tempfile
import os
import time


def test_load_config(tmpdir):
    """
    Test the load_config function by creating a temporary YAML configuration file
    with known content and verifying that the function correctly loads this file into a dictionary.
    """
    sample_config = {"param1": "value1", "param2": "value2"}
    config_file = tmpdir.join("config.yml")
    with config_file.open("w") as f:
        yaml.dump(sample_config, f)

    config_path = str(config_file)
    loaded_config = load_config(config_path)
    assert loaded_config == sample_config, "The configuration was not loaded correctly."


def test_make_edge_list():
    """
    Test the make_edge_list function by providing a sample adjacency list and exit nodes,
    then verifying that the function correctly creates an edge list with appropriate weights.
    """
    adjacent_list = {1: [2], 2: [3]}
    exit_nodes = [3]
    expected_edge_list = {1: {2: 1}, 2: {3: 0}}
    edge_list = make_edge_list(adjacent_list, exit_nodes)
    assert edge_list == expected_edge_list, "Edge list created incorrectly."


# Mock configuration for testing
TEST_CONFIG = {
    "arrival_rate": [0.3],
    "miu_list": {1: 0.5, 2: 0.5},
    "adjacent_list": {0: [1], 1: [2], 2: [3]},
    "buffer_size_for_each_queue": {1: 10, 2: 10},
    "transition_proba_all": {0: {1: 1}, 1: {2: 1}, 2: {3: 1}},
    "max_agents": float("inf"),
    "sim_jobs": 100,
    "entry_nodes": [(0, 1)],
    # Add other necessary parameters for your environment setup
}


@pytest.fixture
def mock_config_file():
    """
    Pytest fixture that creates a temporary configuration file with a predefined set of parameters
    for use in tests that require loading a configuration file.
    """
    with tempfile.NamedTemporaryFile("w", delete=False) as tmpfile:
        yaml.dump(TEST_CONFIG, tmpfile)
        return tmpfile.name


def test_graph_construction(mock_config_file):
    """
    Test the graph construction within the queueing environment by verifying that the
    vertices and edges in the created graph match those expected from a mock configuration.
    """
    q_net = create_queueing_env(mock_config_file)

    # Derive expected vertices and edges from TEST_CONFIG
    expected_vertices = set(TEST_CONFIG["adjacent_list"].keys()) | set(
        [node for sublist in TEST_CONFIG["adjacent_list"].values() for node in sublist]
    )
    expected_edges = [
        (start, end)
        for start, ends in TEST_CONFIG["adjacent_list"].items()
        for end in ends
    ]

    # Verify vertices exist in the graph
    actual_vertices = set(q_net.g.nodes())  # Adjust based on your graph's attribute
    assert (
        expected_vertices == actual_vertices
    ), f"Graph vertices do not match expected. Expected: {expected_vertices}, Actual: {actual_vertices}"

    # Verify edges exist in the graph
    actual_edges = set(q_net.g.edges())  # Adjust based on your graph's attribute
    assert (
        set(expected_edges) == actual_edges
    ), f"Graph edges do not match expected. Expected: {expected_edges}, Actual: {actual_edges}"


def test_environment_attributes(mock_config_file):
    """
    Test the initialization of environment attributes by verifying that attributes of the
    queueing environment match the expected values from the mock configuration file.
    """
    # Create the environment using the mock configuration file
    q_net = create_queueing_env(mock_config_file)

    # Assert that environment attributes match expected values from the configuration
    assert (
        q_net.lamda == TEST_CONFIG["arrival_rate"]
    ), "arrival_rate does not match configuration"
    assert q_net.miu == TEST_CONFIG["miu_list"], "miu_list does not match configuration"
    # Add other assertions as needed for your environment

    # Clean up the temporary file
    os.remove(mock_config_file)


# Mock data assuming load_config returns a dictionary
expected_config = {"param1": "value1", "param2": "value2"}


@patch("foundations.core_functions.load_config", return_value=expected_config)
def test_concurrency(mock_load_config):
    """
    Test function behavior when accessed by multiple threads simultaneously.
    Assumes load_config is a function that loads and returns configuration data.
    """
    # Setup
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(mock_load_config, "config_path") for _ in range(10)]
        results = [f.result() for f in futures]

    # Verify all results are consistent
    assert all(
        r == expected_config for r in results
    ), "Concurrent access should yield consistent results"


def test_state_change():
    "Function to test whether states have changed to avoid actor vector and transition proba diminishing"
    script_dir = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(script_dir))
    config_dir = os.path.join(parent_dir, "user_config")

    # Create the file paths using os.path.join
    config_param_filepath = os.path.join(config_dir, "configuration.yml")
    eval_param_filepath = os.path.join(config_dir, "eval_hyperparams.yml")

    params, hidden = load_hyperparams(eval_param_filepath)
    rl_env = create_simulation_env(params, config_param_filepath)
    agent = create_ddpg_agent(rl_env, params, hidden)

    rl_env.simulate()
    initial_state = rl_env.get_state()
    action = agent.select_action(torch.tensor(initial_state))
    next_state_tensor = torch.tensor(rl_env.get_next_state(action)).float()
    next_state = next_state_tensor.tolist()

    assert next_state != initial_state, "State should have changed"


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_state_change"])
