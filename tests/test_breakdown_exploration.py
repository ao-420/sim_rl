import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

from foundations.breakdown_exploration.breakdown_exploration import *

import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

from evaluation.decision_evaluation.decision_evaluation import *


@pytest.fixture
def mock_rl_env():
    """Fixture to create a mock RL environment object."""

    class MockEnv:
        def reset(self, env):
            pass

    return MockEnv()


@pytest.fixture
def breakdown_engine(mock_rl_env):
    """Fixture to instantiate BreakdownEngine with a mocked RL environment."""
    with patch("builtins.open"), patch(
        "yaml.load",
        return_value={
            "state_exploration_params": {
                "output_json": True,
                "reset": True,
                "output_histogram": True,
                "output_coverage_metric": True,
                "num_sample": 100,
                "w1": 0.5,
                "w2": 0.5,
                "epsilon_state_exploration": 0.1,
                "reset_frequency": 5,
                "num_output": 10,
                "moa_window": 5,
            }
        },
    ):
        engine = BreakdownEngine(mock_rl_env, normal_std=1.0)
        engine.blockage_cases = {
            "example_case": {}
        }  # Direct initialization for testing
        return engine


def test_initialization(breakdown_engine):
    """Test initialization of BreakdownEngine."""
    assert breakdown_engine.std == 1.0
    assert breakdown_engine.device
    assert breakdown_engine.output_json_files


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_save_coverage_metric(mock_show, mock_savefig, breakdown_engine):
    """Test the save_coverage_metric method focuses on file operations."""
    with patch("builtins.open", mock_open()) as mock_file:
        breakdown_engine.save_coverage_metric()
        mock_file.assert_called()  # Check if file is attempted to be opened
        assert (
            not mock_savefig.called
        )  # Ensure that savefig was NOT called, as it's not used in the method

    # To verify that the correct data is being written to the file, you might mock json.dump
    with patch("json.dump") as mock_json_dump:
        breakdown_engine.save_coverage_metric()
        mock_json_dump.assert_called_once()  # Ensure json.dump was called
        # Optionally check for correct data format
        args, kwargs = mock_json_dump.call_args
        expected_data = {
            "coverage": breakdown_engine.num_states_explored
            / len(breakdown_engine.blockage_cases)
        }
        assert args[0] == expected_data  # Validate the data written


def test_create_queue_env(breakdown_engine):
    """Test create_queue_env method with mocking create_params."""
    with patch(
        "foundations.core_functions.create_params",
        return_value=(
            0.1,
            {"1": 1.0},
            ["class1"],
            [{"arg1": "value1"}],
            [],
            [],
            [],
            1,
            100,
            [1],
        ),
    ):
        q_env = breakdown_engine.create_queue_env([1.0])
        assert q_env  # Ensure an object is returned


def test_explore_state(breakdown_engine):
    """Test the state exploration logic."""
    breakdown_engine.blockage_cases = {0: {"case": "data"}}
    case_num, data = breakdown_engine.explore_state()
    assert case_num == 0
    assert data == {"case": "data"}


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
