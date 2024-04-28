import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

from unittest.mock import patch, MagicMock, mock_open

from evaluation.decision_evaluation.decision_evaluation import *

# Setup the test environment
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import pytest
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import matplotlib.pyplot as plt

class TestControlEvaluation(unittest.TestCase):

    @patch('matplotlib.pyplot.savefig')
    def test_plot_queue(self, mock_savefig):
        ce = ControlEvaluation(queue_index=2, metric='throughput')
        ce.fig, ce.ax = plt.subplots()  # Prepare plot
        
        ce.plot_queue(['Test'], [[1, 2, 3]])
        mock_savefig.assert_called_once()  # Ensure plot is saved

    def test_start_evaluation(self):
        ce = ControlEvaluation(queue_index=2, metric='throughput')
        ce.environment = MagicMock()
        ce.environment.net.edge2queue = {2: MagicMock(edge=(0, 1))}
        ce.environment.get_state = MagicMock(return_value='state')
        ce.environment.get_next_state = MagicMock(return_value=('next_state', 'action'))
        ce.environment.return_queue = MagicMock(return_value=10)
        ce.environment.transition_proba = {0: {1: 0.5}}
        
        agent = MagicMock()
        agent.actor = MagicMock(return_value=MagicMock(detach=MagicMock(return_value='action')))
        
        queue_metrics, transition_proba = ce.start_evaluation(agent, 5)
        
        self.assertEqual(len(queue_metrics), 5)
        self.assertEqual(len(transition_proba), 5)

# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
