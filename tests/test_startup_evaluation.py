import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

from evaluation.startup_evaluation.startup_evaluation import *

from unittest.mock import patch, MagicMock
import numpy as np

# Mock data for testing
sample_data = [0.1, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01]


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(sample_data))
@patch("os.path.normpath")
@patch("os.getcwd")
@patch("json.load", return_value=sample_data)
def test_load_json_data(mock_load, mock_getcwd, mock_normpath, mock_file):
    "Test the loading of JSON data"
    mock_getcwd.return_value = "/fake/directory"
    mock_normpath.return_value = (
        "/fake/directory/foundations/output_csv/reward_dict.json"
    )
    app = StartupBehavior(5, 0.01, 3, 0)
    data = app.load_json_data(mock_normpath.return_value)
    assert data == sample_data
    mock_file.assert_called_once_with(
        "/fake/directory/foundations/output_csv/reward_dict.json", "r"
    )
    mock_load.assert_called_once()


def test_calculate_derivative():
    "Test derivative calculation"
    app = StartupBehavior(5, 0.01, 3, 0)
    data = np.array([3, 4, 5, 6, 7])
    expected = np.array([1, 1, 1, 1])
    np.testing.assert_array_almost_equal(app.calculate_derivative(data), expected)


@pytest.mark.parametrize(
    "data, expected",
    [([0.02, 0.01, 0.01, 0.01, 0.01], -1), ([0.02, 0.2, 0.1, 0.05, 0.02], -1)],
)
def test_find_stabilization_point(data, expected):
    "Test finding the stabilization point"
    app = StartupBehavior(5, 0.01, 3, 0)
    result = app.find_stabilization_point(np.array(data))
    assert result == expected


def test_moving_average():
    "Test moving average calculation with different window sizes."
    app = StartupBehavior(3, 0.01, 3, 0)
    data = np.array([1, 2, 3, 4, 5])
    expected = np.array([2, 3, 4])
    np.testing.assert_array_almost_equal(app.moving_average(data), expected)


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
