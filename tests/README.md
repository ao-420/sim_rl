# Pytest-Driven Unit Testing for Simulation-Driven RL

This document provides an overview of the test suites for repository of simulation-drive RL. Each test file targets specific functionalties within the simulation, ensuring reliability and smoothy performance.

## Testing Files Overview

### test_agents.py

Tests the functionalities of various agent implementations, ensuring correct initialization, action selection, and learning updates. Key areas covered include:

- Initialization of DDPG (Deep Deterministic Policy Gradient) agents with specific configurations.
- Proper functioning of model fitting, including loss calculation and optimization.
- Verification of action selection based on given states.
- Q-value updates and their impact on learning.

### test_buffer.py

Focuses on the Replay Buffer's capacity management, sampling, and edge case handling. It verifies:

- Correct handling of buffer capacity, ensuring oldest entries are discarded when the buffer is full.
- The sampling mechanism returns a sample of the requested size and structure.
- Handling of edge cases, such as sampling from an empty buffer or requesting more samples than available.

### test_model.py

Ensures the integrity of various neural network models, including Actor, Critic, RewardModel, and NextStateModel. Tests check:

- Output shapes of models against expected dimensions.
- Range and validity of the Actor model outputs.
- Proper construction and error handling of model input parameters.

### test_queueing_network.py

Validates the functionality of the queueing network, from configuration processing to network initialization. Areas tested include:

- Correct parsing and application of configuration parameters from YAML files.
- Initialization of the queue network with correct attributes and operational parameters.

### test_RL_Environment.py

Tests the Reinforcement Learning Environment setup, including initialization, state management, and reward calculation. It focuses on:

- Correct initialization based on different configuration files.
- Proper calculation and assignment of entry and exit nodes.
- State transitions and action effect validation.
- Reward signal accuracy based on environment dynamics.

### test_state_exploration.py

Examines state exploration mechanisms, particularly in relation to DDPG agents. It verifies:

- Correct loading and application of hyperparameters.
- Proper ranking of states based on Q-values, ensuring the exploration strategy is as expected.

### test_supporting_functions.py

Ensures utility and supporting functions work as intended. Tests cover:

- Configuration loading from YAML files.
- Correct calculation of connections within network models.
- Edge list creation and validation against expected structures.

## Running the Tests

Step 1: Open a terminal and navigate to the root directory of your tests. For example: "D:\MScDataSparqProject\tests"

Step 2: Run the pytest command without specifying any particular file:
```bash
pytest
```
This will automatically run all test files in the current directory.