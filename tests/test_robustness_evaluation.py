import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

from evaluation.robustness_evaluation.robustness_evaluation import *

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from scipy.stats import norm


@pytest.fixture
def num_runs():
    """Fixture to create a NumRuns instance with pre-configured parameters."""
    return RobustnessEvaluation(
        confidence_level=0.95, desired_error=1, num_runs=10, time_steps=100, num_sim=100
    )


def test_initialization(num_runs):
    """Test initialization of the NumRuns class."""
    assert num_runs.confidence_level == 0.95
    assert np.isclose(num_runs.z_value, norm.ppf((1 + 0.95) / 2))
    assert num_runs.desired_error == 1
    assert num_runs.num_runs == 10


def test_get_std(num_runs):
    """Test the standard deviation calculation."""
    with patch.object(
        num_runs,
        "train_multi_agents",
        return_value=[
            {"key": {"inner_key": np.random.normal(0, 1, 100)}} for _ in range(10)
        ],
    ):
        std_devs = num_runs.get_std("fake_config.yml", "fake_eval.yml")
        assert isinstance(std_devs, np.ndarray)
        assert std_devs.shape[1] == 100  # Assuming flattened array size


def test_get_req_runs(num_runs):
    """Test calculation of required number of runs based on standard deviations."""
    num_runs.std_devs = np.array([1.0])  # Directly assign an array to std_devs
    required_runs = num_runs.get_req_runs()
    expected_runs = (num_runs.z_value * 1.0 / num_runs.desired_error) ** 2
    assert round(1 + required_runs) == round(1 + expected_runs)


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
