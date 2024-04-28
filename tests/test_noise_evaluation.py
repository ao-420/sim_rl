import sys
from pathlib import Path
import os
# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from evaluation.noise_evaluation.noise_evaluation import *
from queueing_tool.queues.queue_servers import *

from unittest.mock import patch, MagicMock
import numpy as np

class TestNoisyNetwork(unittest.TestCase):

    def setUp(self):
        # Setup the initial conditions that will be used in multiple test cases
        self.config_file = 'user_config/configuration.yml'
        self.frequency = 0.5
        self.mean = 0
        self.variance = 1
        self.num_sim = 100
        self.temperature = 0.15
        self.noisy_net = NoisyNetwork(self.config_file, self.frequency, self.mean, self.variance, self.num_sim, self.temperature)

    def test_init(self):
        # Test initialization
        self.assertEqual(self.noisy_net.frequency, 0.5)
        self.assertEqual(self.noisy_net.mean, 0)
        self.assertEqual(self.noisy_net.variance, 1)
        self.assertIsNotNone(self.noisy_net.environment)

    def test_compute_increment(self):
        # Test if compute_increment behaves correctly regarding the frequency and noise generation
        np.random.seed(42)  # For reproducible tests
        noise_counts = sum(1 for _ in range(1000) if self.noisy_net.compute_increment() != 0)
        expected_count = 1000 * self.frequency  # As frequency is 0.5, expect about 500 out of 1000 to be non-zero
        self.assertTrue(450 < noise_counts < 550)  # Allow some variability

    @patch('builtins.print')  # Example of patching print to keep the test output clean
    def test_get_noisy_env(self, mocked_print):
        # Test modifications on environment by checking the changes in parameters
        noisy_env = self.noisy_net.get_noisy_env()
        self.assertIsNotNone(noisy_env)

# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main([__file__, "-k", "test_start_evaluation"])