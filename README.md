
# RL-Diven Queueing Network Simulation

This repository implements a Dyna-DDPG (Deep Deterministic Policy Gradient) Reinforcement Learning agent that optimizes routing probabilities to maximize End-to-End (EtE) delay and throughput in a simulated queueing network.

## Project Structure

- `agents`: Contains the Dyna-DDPG agent implementation and allows the integration of new types of agents for exploring the simulated queueing environment.
- `queue_env`: Defines the simulated queueing environment, utilizing functionalities from the `queueing-tool` package.
- `rl_env`: Hosts the RL environment, which is portable and compatible with different agent types.
- `features`: Includes several utility features:
  - **Decision Evaluation**: Demonstrates how the agent responds to a server outage by adjusting routing probabilities.
  - **Convergence Evaluation**: Assesses the stability and reliability of the agent across different training setups
  - **Noise Evaluation**: Evaluate the effect of environmental noise on the performance of the agent
  - **Startup Evaluation**: Identifies the burn-in period of the agent
  - **Robustness Evaluation**: Assess robustness of decisions across multiple trained agents

## Prerequisites

Before running the simulations, ensure you have the following installed:
- Python >=3.10 <3.12
- torch = "2.2.0"
- numpy = "1.26.4"
- pandas = "2.2.0"
- queueing_tool = "1.2.5"
- matplotlib = "3.8.3"
- wandb = "0.16.3"
- PyYAML = "6.0.1"
- ray = { version = "2.9.2", extras = ["train", "tune"] }
- tqdm = "4.57.0"
- scipy = "1.12.0"

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/ao-420/sim_rl.git
cd sim_rl
pip install -r requirements.txt
```
To view package documentation, run the following command in the root folder (sim_rl):

```bash
open docs/_build/html/index.html
```

## Step 1: Configuration

### Environment Setup

#### **Queueing Environment Configuration**

The simulation environment requires the following parameters to be defined in the `configuration.yml`.

- `adjacent_list`: A dictionary defining the adjacency list for the network topology.
- `miu_dict`: A dictionary of service rates for each service node in the network.
- `transition_proba_all`: A dictionary defining the transition probabilities between nodes.
- `active_cap`: The active capacity of the nodes from outside the network.
- `deactive_t`: The deactivation threshold for the nodes from outside the network.
- `buffer_size_for_each_queue`: A dictionary that defines the buffer size for each queue.
- `arrival_rate`: A list that defines the arrival rates for all possible entry nodes.
- `max_agents`: A value that defines the maximum number of agents can be accpeted from outside the network for the entry nodes.
- `sim_jobs`: A value that defines the number of jobs being simulated during every simulation.
- `max_arr_rate_list`: A list that defines the maximum arrival rate for all entry queues.
- `entry_nodes`: A list that defines the source and target vertices of each entry node.

   Example:
   ```yaml
   miu_list:  
   1: 0.250
   2: 0.25
   3: 0.01500
   4: 100
   5: 1.20
   6: 0.01000
   7: 10
   8: 0.1000
   9: 0.500

   adjacent_list:
   0: [1]
   1: [2, 3, 4]
   2: [5]
   3: [6, 7]
   4: [8]
   5: [9]
   6: [9]
   7: [9]
   8: [9]
   9: [10]

   buffer_size_for_each_queue: 
   0: 5000
   1: 5000
   2: 5000
   3: 5000
   4: 5000
   5: 5000
   6: 5000
   7: 5000
   8: 5000
   9: 5000
   10: 5000
   11: 5000
   12: 5000

   transition_proba_all:
   0: {1: 1}
   1: {2: 0.33, 3: 0.33, 4: 0.34}
   2: {5: 1}
   3: {6: 0.5, 7: 0.5}
   4: {8: 1}
   5: {9: 1}
   6: {9: 1}
   7: {9: 1}
   8: {9: 1}
   9: {10: 1}
   
   active_cap: 5

   deactive_t: 0.12

   arrival_rate: [0.3]

   max_agents: inf

   sim_jobs: 100

   max_arr_rate_list: [0.3]

   entry_nodes:
   - [0, 1] 
   ```

#### **RL Environment Parameters**

Set up the RL environment parameters in `eval_hyperparams.yml`:

- `num_episodes`: The number of episodes to run the simulation.
- `num_epochs`: The number of epochs for training.
- `time_steps`: The number of time steps in each episode.
- `batch_size`: Size of the batch used in training. (Default is equal to time_steps)
- `num_sim`: The number of simulations to run during training.
- `tau`: Coefficient for soft update of the target parameters.
- `actor_lr`: Learning rate for the Actor network optimizer.
- `critic_lr`: Learning rate for the Critic Network optimizer.
- `discount`: Discount factor for future rewards.
- `planning_steps`: The number of steps during planning.
- `planning_std`: Standard deviation of the normal disturbance during planning.
- `actor_network`: Network architecture for actor network.
- `critic_network`: Network architecture for critic network.
- `reward_model`: Network architecture for reward model usd in planning.
` `next_state_model`: Network architecture for next state model used in planning. 

   Example:
   ```yaml
   num_episodes: 5

   threshold: 10

   num_epochs: 100

   time_steps: 30

   batch_size: 30
   
   target_update_frequency: 100
   
   buffer_size: 10000
   
   num_sim: 10
   
   tau: 0.5

   num_train_AC: 10
   
   critic_lr: 0.01

   actor_lr: 0.0001
   
   discount: 0.8
   
   planning_steps: 10
   
   planning_std: 0.1

   account_for_blockage: False

   actor_network:
   - 64
   - 64
   - 64

   critic:
   - 64
   - 64
   - 64

   reward_model:
   - 32
   - 64
   - 64
   - 32

   next_state_model:
   - 32
   - 64
   - 64
   - 32
   ```

#### **Tuning Configuration**
Set up the hyperparameter tuning ranges in `tuning_params.yml`:

- `lr_min/max`: Min and max ranges of the learning rate being tuned.
- `epochs_list`: A list that defines the range of possible epochs to train reward model and next state model.
- `batch_size`: A list that defines the range of batch sizes to sample from the replay buffer.
- `tau_min/max`: Min and max ranges of the soft update parameters.
- `discount min/max`: Min and max ranges of the discount factor for future rewards.
- `epsilon_min/max`: Min and max ranges of the standard deviation of normal disturbances during planning.
- `planning_steps`: A list that defines possible steps for planning.
- `w1/w2`: Weight parameters that influence the exploration between key and peripheral states.
- `num_episodes`: A list that defines the possible numbers of episodes to train the agents.
- `time_steps`: A list that defines the possible number of time steps during each episode

   Example:
   ```yaml
   learning_rate_max: 0.1
   learning_rate_min: 0.001

   epochs_list:
   - 10
   - 10
   - 10

   batch_size:
   - 16
   - 32
   - 64

   tau_min: 0.0005
   tau_max: 0.002

   discount_min: 0.1
   discount_max: 0.3

   epsilon_min: 0.1
   epsilon_max: 0.3

   planning_steps: 
   - 10

   num_sample: 
   - 50

   w1: 
   - 0.5

   w2: 
   - 0.5

   num_episodes: 
   - 5

   time_steps: 
   - 10
   
   num_train_AC: 
   - 10

   ```

## Step 2: Running Simulations

### Training Agent
This command starts training the agent within the simulated queueing environment. Results are saved in `/foundations/output_csv` and `/foundations/output_plots`.

```bash
python main.py --function train --config_file user_config/configuration.yml --param_file user_config/eval_hyperparams.yml --data_file output_csv --image_file output_plots --plot_curves True --save_file True
```

### Hyperparameter Tuning

Below provides users two types of tuning strategies that feature different functionalities.

#### **Wandb Tuning**

A machine learning development platform that allows users to track and visualize varou aspects of their model training process in real-time, including loss and accuracy charts, parameter distributions, gradient histograms and system metrics. To run wandb:

```bash
python main.py --function tune --config_file user_config/configuration.yml --param_file user_config/eval_hyperparams.yml --data_file output_csv --image_file output_plots --plot_curves True --save_file True --tuner wandb 
```

#### **Ray Tuning**

An industry standard tool for distributed hyperparameter tuning which integrates with TensorBoard and extensive analysis libraries. It also allows users to leverage cutting edge optimization algorithms at scale, including Bayesian Optimization, Population Based Training and HyperBand. To run ray tuning:

```bash
python main.py --function tune --config_file user_config/configuration.yml --param_file user_config/eval_hyperparams.yml --data_file output_csv --image_file output_plots --plot_curves True --save_file True --tuner ray_tune
```

## Step 3: Explore Features

### 1. **Explore Breakdown Scenarios**

This feature allows the user to train the agent based on customed exploration preferences between key states and peripheral states using weight parameter `w1_key` and `w2_peripheral`. The purpose of this feature is to enable the agent to not only generate high rewards for key states but also visit all breadown scenarios sufficiently enough. 

Set up the parameters in `user_config\features_params\bloackage_explore_params.yml`:

- `w1_key`: Weight parameter to control favor exploring key states.
- `w2_peripheral`: Weight parameter to control favor exploring peripheral states.
- `reset`: A bool value that controls whether to reset weight parameters during training.
- `reset_frequency`: A value that defines the number of episodes frequency to reset the weight parameters.
- `num_output`: A value that defines the number of top and least reward/visits states to plot in a histogram
- `output_json`: A bool value that determines whether to output the json file of key states and peripheral states
- `output_histogram`: A bool value that determines whether to output the histogram that shows the rewards and visits of the top and least states.
- `output_coverage_metric`: A bool value that determines whether to output the current coverage metric.

To run this feature, navigate to `/foundations/breakdown_exploration` and run:
   ```bash
   python breakdown_exploration.py
   ```

### 2. **Decision Evaluation (Blockage Demonstrations)**

This feature allows the user to test a trained agent's performance on a simulated server blockage queueing environment by visualizing the changes in transition probabilities. The purpose of this feature is to show how effectice the tranied agent is acting on breakdown cases. 

- `num_sim`: Defines the number of jobs to simulate for each time step during training.
- `time_steps`: Defines the number of time steps to perform for each episode.
- `queue_index`: Defines the queue index that record the metrics for.
- `metric`: Defines the metric to be reported for the selected queue.

To use this feature, navigate to `/evaluation/decision_evaluation` and run:
   ```bash
   python decision_evaluation.py
   ```

### 3. **Startup Behavior Identification**

This feature allows the user to visualize when the burn-in periods end on the learning curve. 

Set up the parameters in the script:

- `window_size`: Specifies the number of data points used to compute the moving average of the rewards.
- `threshold`: Defines the maximum acceptable absolute value of the derivative of the smoothed rewards below which a reward is considered stable. 
- `consecutive_points`: The number of consecutive data points that must all be below the threshold for the rewards to be considered as having stabilized. 
- `episode`: Specify which episode's rewards to analyze from a dataset.

To perform the feature, navigate to `/evaluation/startup_evaluation` and run:
   ```bash
   python startup_evaluation.py
   ```

### 4. **Convergence Evaluation** 

This feature allows the user train multiple versions of the agent for different numbers of training episodes and then evaluate the performance of each agent on the simulation environment.

Set up the parameters in the script:

- `num_episodes_list`: A list that contains different numbers of episodes to train the agents.
- `timesteps`: A value that defines the number of timesteps to train the agent during each episode.

To run this feature, navigate to `/evaluation/convergence_evaluation` and run:
   ```bash
   python convergence_evaluation.py
   ```

### 5. **Robustness Evaluation** 

This feature allows the user to train multiple agents, analyze their behavior, and calculate statistical metrics based on their performance.

Set up the parameters in the script:

- `confidence_level`: The statistical confidence level for calculations.
- `desired_error`: The target error margin for estimating statistical requirements.
- `num_runs`: Number of times to train agents.
- `time_steps`: Number of time steps each agent runs in the simulation environment.
- `num_sim`: Number of simulations to run in the environment.

To run this feature, navigate to `/evaluation/robustness_evaluation` and run:
   ```bash
   python robustness_evaluation.py
   ```

### 6. **Noise Evaluation** 

This feature allows the user to evaluate the effect of environmental noise on the performance of the agent.

Set up the parameters in the script:

- `frequency `: The likelihood or frequency at which noise is introduced to the system. It must be a value between 0 and 1. This parameter determines how often, proportionally, noise will be added during the simulation. 
- `mean`: The mean of the normal distribution from which the noise values are sampled. This represents the average value of the noise that will be introduced.
- `variance`: The variance of the normal distribution from which the noise values are sampled. This parameter indicates the spread or dispersion of the noise around the mean.

To run the feature, navigate to `/evaluation/noise_evaluation` and run:
   ```bash
   python noise_evaluation.py
   ```

## Contribution

Contributions are welcome. Please create a pull request or issue to discuss proposed changes or report bugs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
