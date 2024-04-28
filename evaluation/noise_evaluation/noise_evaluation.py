# This class will be used to evaluate the effect of environmental noise on the performance of the agent
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
from foundations.core_functions import Engine
import numpy as np
import copy
import os
from queue_env.queueing_network import Queue_network
from queue_env.queue_base_functions import *


# Definition of the Noisy Network class variant
class NoisyNetwork:
    def __init__(self, config_file, frequency, mean, variance, num_sim=100, temperature = 0.15):
        """
        Args:
            frequency(float ): the frequency at which noise is added to the environment - enforce that its between 0 and 1
            mean (float): Mean of the distribution from which the noise is sampled
            variance (float): Variance of the distribution from which the noise is sampled
        """
        self.frequency = frequency
        self.mean = mean
        self.variance = variance
        self.num_sim = num_sim
        self.temperature = temperature
        self.environment = create_simulation_env({"num_sim": self.num_sim, 'temperature': self.temperature}, config_file)
        self.config_params = load_config(config_file)

    def compute_increment(self):
        """This function is main entry point for adding noise to the environment. This function samples from a normal distribution with mean and variance specified in the constructor and
        returns the noise increment to be added to the environment with a probability specified by the frequency parameter.
        Args:

        """
        if self.frequency > np.random.random():
            # Determines whether we are currently at a noise injection interval
            noise = np.random.normal(self.mean, self.variance)
            return noise
        else:
            return 0

    def get_noisy_env(self):

        q_args = self.environment.qn_net.q_args
        entry_node_encountered = 0
        for edge_type in q_args.keys():
            if "arrival_f" in q_args[edge_type].keys():
                max_arrival_rate = self.config_params["arrival_rate"][
                    entry_node_encountered
                ]
                rate = (
                    lambda t: 0.1 * (max_arrival_rate)
                    + (1 - 0.1) * (max_arrival_rate) * np.sin(np.pi * t / 2) ** 2
                )
                # the noise is added to the arrival rate here
                q_args[edge_type]["arrival_f"] = (
                    lambda t, rate=rate: poisson_random_measure(
                        t, rate, max_arrival_rate
                    )
                    + self.compute_increment()
                )
                q_args[edge_type]["noise"] = lambda: self.compute_increment()
                entry_node_encountered += 1

        org_net = self.environment.qn_net
        new_net = copy.copy(org_net)
        new_net.process_input(
            org_net.lamda,
            org_net.miu,
            org_net.q_classes,
            q_args,
            org_net.adja_list,
            org_net.edge_list,
            org_net.transition_proba,
            org_net.max_agents,
            org_net.sim_jobs,
        )
        new_net.create_env()
        noisy_environment = RLEnv(qn_net=new_net, num_sim=5000)
        return noisy_environment

    def train(
        self, params, agent, env, save_file=True, data_filename="output_csv_noisy"
    ):

        (
            next_state_model_list_all,
            critic_loss_list,
            actor_loss_list,
            reward_by_episode,
            action_dict,
            gradient_dict,
            transition_probas,
        ) = train(params, agent, env)

        evaluation_dir = "evaluation"
        noise_dir = "noise_evaluation"
        csv_filepath = os.path.join(root_dir, evaluation_dir, noise_dir, data_filename)

        if save_file:
            save_all(
                next_state_model_list_all,
                critic_loss_list,
                actor_loss_list,
                reward_by_episode,
                action_dict,
                gradient_dict,
                transition_probas,
                output_dir=csv_filepath,
            )

    def start_evaluation(self, noisy_env=None, agent=None, time_steps=100):
        if noisy_env is None:
            noisy_env = self.get_noisy_env()
        if agent is None:
            agent_path = "agents"
            agent = "trained_agent.pt"
            path_to_saved_agent = os.path.join(root_dir, agent_path, agent)
            agent = torch.load(path_to_saved_agent)
        reward = start_evaluation(noisy_env, agent, time_steps)
        print(f"Total reward on environment with external noise is:{reward}")
        return reward


# Running the code for the noise evaluation
if __name__ == "__main__":

    frequency = 0.5
    mean = 0
    variance = 1
    timesteps = 100
    temperature = 0.15

    # # Define the object of the NoiseEvaluator class
    config_file = "user_config/configuration.yml"
    eval_file = "user_config/eval_hyperparams.yml"
    noisy_net = NoisyNetwork(config_file, frequency, mean, variance, temperature)
    noisy_env = noisy_net.get_noisy_env()

    # # When introducing noise in the training we call the start_train method of the NoiseEvaluator object
    params, hidden = load_hyperparams(eval_file)
    agent = create_ddpg_agent(noisy_env, params, hidden)
    noisy_net.train(params, agent, noisy_env)
    # noise_evaluator.start_train(eval_env, agent,save_file = True, data_filename = 'output_csv', image_filename = 'output_plots')

    # # When introducing noise in the the control of the control of the environment we first define the agent
    agent_path = "agents"
    agent = "trained_agent.pt"
    path_to_saved_agent = os.path.join(root_dir, agent_path, agent)
    saved_agent = torch.load(path_to_saved_agent)
    noisy_net.start_evaluation(noisy_env, saved_agent, time_steps=100)
    # noise_evaluator.start_evaluation(eval_env , saved_agent,timesteps)
