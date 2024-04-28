import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))
import torch
import matplotlib.pyplot as plt
from rl_env.RL_Environment import *
from queue_env.queueing_network import *
from foundations.core_functions import *
from scipy.stats import norm
import numpy as np
import copy
import os

# Required file paths
agent_filepath = os.path.join(root_dir, "agents/trained_agent.pt")
config_param_filepath = "user_config/configuration.yml"
eval_param_filepath = "user_config/eval_hyperparams.yml"
data_filename = "output_csv"
image_filename = "output_plots"


# Training Multiple Agents
class RobustnessEvaluation:
    def __init__(
        self,
        confidence_level=0.95,
        desired_error=1,
        num_runs=10,
        time_steps=100,
        num_sim=100,
        temperature = 0.15
    ):
        """
        This class is designed to handle the training of multiple agents to estimate
        statistical properties of their performance.

        Parameters:
            confidence_level (float): The confidence level for the error estimation - defaults to 0.95.
            desired_error (int): The desired margin of error for the estimation - defaults to 1.
            num_runs (int): The number of different agents to train - defaults to 10.
            time_steps (int): The number of time steps to run for each simulation - defaults to 100.
            num_sim (int): The number of simulations to average for statistical estimation - defaults to 100.
        """
        self.agents = None
        self.num_runs = num_runs
        self.time_steps = time_steps
        self.confidence_level = confidence_level
        self.z_value = norm.ppf((1 + confidence_level) / 2)
        self.desired_error = desired_error
        self.num_sim = num_sim
        self.temperature = temperature

    def train_multi_agents(self, config_param_filepath, eval_param_filepath):

        for run in range(self.num_runs):
            start_train(
                config_file=config_param_filepath,
                param_file=eval_param_filepath,
                data_filename=data_filename,
                image_filename=image_filename,
                save_file=False,
                plot_curves=False,
            )
            if self.agents == None:
                self.agents = []
                self.agents.append(torch.load(agent_filepath))
            else:
                self.agents.append(torch.load(agent_filepath))

        agents_transition_proba = []

        for agent in self.agents:
            env = create_simulation_env(
                {"num_sim": self.num_sim, 'temperature': self.temperature}, config_param_filepath
            )
            for time_step in range(self.time_steps):
                state = env.get_state()
                action = agent.actor(state).detach()
                state = env.get_next_state(action)[0]
            agents_transition_proba.append(env.transition_proba)
        return agents_transition_proba

    def get_std(self, config_param_filepath, eval_param_filepath):

        agents_transition_proba = self.train_multi_agents(
            config_param_filepath, eval_param_filepath
        )
        transition_proba_all_flattened = []

        for agent_proba_dict in agents_transition_proba:
            transition_proba_values = [
                np.array(list(value_dict.values()), dtype=float)
                for value_dict in agent_proba_dict.values()
                if value_dict
            ]
            transition_proba_values = np.concatenate(transition_proba_values)
            transition_proba_all_flattened.append(transition_proba_values)
        print(transition_proba_all_flattened)
        std_devs = np.std(transition_proba_all_flattened, axis=0)
        self.std_devs = std_devs
        return std_devs

    def get_req_runs(self):
        self.estimated_std_dev = np.max(self.std_devs)
        required_n = (self.z_value * self.estimated_std_dev / self.desired_error) ** 2
        print("Required number of runs:", round(1 + required_n))
        return required_n


if __name__ == "__main__":
    # Example usage
    # 1. Set up the parameters for the robustness evaluation
    confidence_level = 0.95
    desired_error = 1
    num_runs = 10
    time_steps = 100
    num_sim = 100

    # 2. Initialize the robustness evaluation class
    nr = RobustnessEvaluation(
        confidence_level=confidence_level,
        desired_error=desired_error,
        num_runs=num_runs,
        time_steps=time_steps,
        num_sim=num_sim,
    )

    # 3. Train multiple agents and obtain the transition probabilities
    std = nr.get_std(
        config_param_filepath=config_param_filepath,
        eval_param_filepath=eval_param_filepath,
    )
    nr.get_req_runs()
