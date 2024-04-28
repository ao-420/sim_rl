import sys
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
parent_dir = os.path.dirname(os.path.dirname(root_dir))
os.chdir(parent_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
from agents.ddpg_agent import DDPGAgent
import torch
from queue_env.queueing_network import Queue_network
from foundations.core_functions import *
import json
import matplotlib.pyplot as plt
import yaml
import matplotlib.pyplot as plt


class BreakdownEngine:
    def __init__(self, rl_env, normal_std):
        """
        Initialize the ExploreStateEngine with default parameters and configurations.
        """
        self.eval_param_filepath = "user_config/eval_hyperparams.yml"
        self.config_filepath = "user_config/configuration.yml"
        self.rl_env = rl_env

        self.activate_features()
        self.load_params()
        self.init_device()

        self.std = normal_std

    def activate_features(self):
        """
        Activate features based on loaded hyperparameters.
        """
        params = self.load_hyperparams()

        self.output_json_files = params["output_json"]
        self.reset = params["reset"]
        self.output_histogram = params["output_histogram"]
        self.output_coverage_metric = params["output_coverage_metric"]

    def init_device(self):
        """
        Initialize the computation device (CPU or CUDA) for PyTorch operations.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_hyperparams(self):
        """
        Load hyperparameters from a YAML file.

        Parameters:
        - param_filepath (str): The file path to the hyperparameters YAML file.

        Returns:
        - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
        """

        # Assuming __file__ is somewhere inside 'D:\\MScDataSparqProject'
        project_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        # Now directly append your target directory to the project base
        abs_file_path = os.path.join(project_dir, "user_config", "eval_hyperparams.yml")

        with open(abs_file_path, "r") as env_param_file:
            parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)
        params = parameter_dictionary["state_exploration_params"]

        return params

    def get_param_for_state_exploration(self, params):
        """
        Extract parameters necessary for state exploration.

        Parameters:
        - params (dict): Hyperparameters including those needed for state exploration.

        Returns:
        - tuple: A tuple containing parameters specific to state exploration.
        """
        self.num_sample = params["num_sample"]
        self.w1 = params["w1"]
        self.w2 = params["w2"]
        self.epsilon = params["epsilon_state_exploration"]
        if self.reset == False:
            self.reset_frequency = None
        else:
            self.reset_frequency = params["reset_frequency"]
        self.num_output = params["num_output"]
        self.moa_coef = params["moa_window"]

        self.num_states_explored = 0
        self.states_with_rewards = {}
        self.states_with_visits = {}

    def load_params(self):
        """
        Load parameters for state exploration from the hyperparameters file.
        """
        params = self.load_hyperparams()
        self.get_param_for_state_exploration(params)

    def reset_weights(self, episode, reset_frequency):
        """
        Reset the weights based on the episode and frequency.

        Parameters:
        - episode (int): The current episode number.
        - reset_frequency (int): The frequency at which weights should be reset.
        """
        if episode != 0 and episode % reset_frequency == 0:
            print()
            weights = input(
                f"{episode} episodes have passed. Please reset your weights:"
            )
            weights = [float(x) for x in weights.split(",")]
            self.w1, self.w2 = weights

    def create_queue_env(self, miu_list):
        """
        Create and configure a queueing environment based on a given configuration file.

        Parameters:
        - config_file (str): The file path to the environment configuration file.

        Returns:
        - Queue_network: An instance of the queueing environment.
        """
        (
            arrival_rate,
            miu_dict,
            q_classes,
            q_args,
            adjacent_list,
            edge_list,
            transition_proba_all,
            max_agents,
            sim_jobs,
            entry_nodes,
        ) = create_params(self.config_filepath, miu_list)

        q_net = Queue_network()
        q_net.process_input(
            arrival_rate,
            miu_dict,
            q_classes,
            q_args,
            adjacent_list,
            edge_list,
            transition_proba_all,
            max_agents,
            sim_jobs,
        )
        q_net.create_env()
        return q_net

    def create_blockage_cases(self):
        """
        Creates blockage cases based on 'miu' values from the network model,
        setting a blockage (high value) on all nodes except the first one.

        Returns:
            dict: A dictionary where keys are state names and values are dictionaries
                representing 'miu' values for each state, modified to simulate blockages.
        """
        miu_dict = self.rl_env.qn_net.miu

        blockage_cases = {}
        for key in miu_dict.keys():
            new_miu_dict = {key: miu_dict[key] for key in miu_dict.keys()}
            if key != 1:
                new_miu_dict[key] = float("inf")

                key_name = f"bn_{key}"
                blockage_cases[key_name] = new_miu_dict

        relative_path = "foundations\\breakdown_exploration\\output_data"

        output_file_path = os.path.join(
            root_dir, relative_path, "all_breakdown_cases.json"
        )
        with open(output_file_path, "w") as json_file:
            json.dump(blockage_cases, json_file)

        print(f"All breakdown cases have been saved at {output_file_path}")

        return blockage_cases

    def get_reward(self):
        """
        Calculates the average reward across all episodes.

        Returns:
            float: The mean of average rewards from all episodes.
        """
        rewards = []
        for eps in self.reward_by_episode.keys():
            rewards.append(np.mean(self.reward_by_episode[eps]))

        return np.mean(rewards)

    def save_coverage_metric(self):
        """
        Computes and saves the coverage metric as the ratio of explored states
        to the total number of blockage cases to a JSON file.

        The path for the saved file is relative to the current working directory.
        """
        coverage = {}
        coverage["coverage"] = self.num_states_explored / len(self.blockage_cases)

        current_path = os.getcwd()
        relative_path = "features\\state_exploration\\output_data\\coverage_metric"

        output_file_path = os.path.join(current_path, relative_path, "coverage.json")

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open(output_file_path, "w") as json_file:
            json.dump(coverage, json_file, indent=4)

        print(f"Coverage Metric has been saved at {output_file_path}")

    def get_sorted_key_states(self):
        for case_num, _ in enumerate(self.blockage_cases):
            if case_num not in self.states_with_rewards.keys():
                self.states_with_rewards[case_num] = -float("inf")

        sorted_states = sorted(
            self.states_with_rewards.items(), key=lambda item: item[1], reverse=False
        )
        return sorted_states

    def save_key_states(self):
        """
        Processes and saves information about key states based on rewards.
        This includes sorting states by rewards, saving this data to a JSON file,
        and plotting a histogram of the rewards.

        Returns:
            list: A list of tuples containing state indices and their corresponding rewards, sorted by rewards.
        """

        sorted_states = self.get_sorted_key_states()

        keys, values = zip(*sorted_states)

        plt.figure(figsize=(10, 5))
        plt.bar(keys, values, color="blue")
        plt.xlabel("Sorted States")
        plt.ylabel("Rewards")
        plt.title("States Ranked by Rewards")
        plt.xticks(ticks=range(len(keys)), labels=keys)

        current_path = os.getcwd()
        relative_path = "features\\state_exploration\\output_data\\key_states"

        output_plot_path = os.path.join(current_path, relative_path, "rewards.png")
        output_file_path = os.path.join(current_path, relative_path, "key_states.json")

        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        plt.savefig(output_plot_path)

        with open(output_file_path, "w") as json_file:
            json.dump(sorted_states, json_file)

        print(f"Key states and their reward plots have been saved at {relative_path}")

        return sorted_states

    def get_sorted_peripheral_states(self):
        for case_num, _ in enumerate(self.blockage_cases):
            if case_num not in self.states_with_visits.keys():
                self.states_with_visits[case_num] = 0

        sorted_states = sorted(
            self.states_with_visits.items(), key=lambda item: item[1], reverse=False
        )
        return sorted_states

    def save_peripheral_states(self):
        """
        Processes and saves information about peripheral states based on visits.
        This includes sorting states by visits, saving this data to a JSON file,
        and plotting a histogram of the visits.

        Returns:
            list: A list of tuples containing state indices and their corresponding visits, sorted by visits.
        """
        sorted_states = self.get_sorted_peripheral_states()

        keys, values = zip(*sorted_states)

        plt.figure(figsize=(10, 5))
        plt.bar(keys, values, color="blue")
        plt.xlabel("Sorted States")
        plt.ylabel("Visits")
        plt.title("States Ranked by Visits")
        plt.xticks(ticks=range(len(keys)), labels=keys)

        current_path = os.getcwd()

        relative_path = "features\\state_exploration\\output_data\\peripheral_states"

        output_plot_path = os.path.join(current_path, relative_path, "visits.png")
        output_file_path = os.path.join(
            current_path, relative_path, "peripheral_states.json"
        )

        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

        plt.savefig(output_plot_path)

        with open(output_file_path, "w") as json_file:
            json.dump(sorted_states, json_file)

        print(
            f"Peripheral states and their visits plots have been saved at {relative_path}"
        )

        return sorted_states

    def explore_state(self):
        """
        Explores states by weighting and combining rewards and visits data,
        sorts them to identify the most significant state.

        Returns:
            tuple: The most significant case number and its corresponding blockage case data.
        """
        sorted_key_states = self.get_sorted_key_states()
        sorted_peripheral_states = self.get_sorted_peripheral_states()

        sorted_peripheral_states_by_index = [
            (x[0], i + 1) for i, x in enumerate(sorted_peripheral_states)
        ]
        sorted_key_states_by_index = [
            (x[0], i + 1) for i, x in enumerate(sorted_key_states)
        ]

        result = [
            (x[0], self.w1 * x[1] + self.w2 * y[1])
            for x, y in zip(
                sorted_peripheral_states_by_index, sorted_key_states_by_index
            )
        ]
        sorted_result = sorted(result, key=lambda x: x[1], reverse=False)

        case_num, _ = sorted_result[0]

        return case_num, self.blockage_cases[case_num]

    def run(self, agent):
        """
        Runs the simulation for a specified number of iterations using a given agent.
        Each run involves resetting environments, creating blockage cases, exploring states,
        and updating state statistics.

        Args:
            agent: The agent responsible for running simulations.
            num_runs (int): Number of simulation runs to perform.
        """
        self.blockage_cases_dict = self.create_blockage_cases()
        self.blockage_cases = list(self.blockage_cases_dict.values())

        for run in range(len(self.blockage_cases)):

            if self.reset:
                self.reset_weights(run, self.reset_frequency)

            case_num, miu_list = self.explore_state()

            queue_env = self.create_queue_env(miu_list)
            self.rl_env.reset(queue_env)

            (
                self.next_state_model_list_all,
                self.critic_loss_list,
                self.actor_loss_list,
                self.reward_by_episode,
                self.action_dict,
                self.gradient_dict,
                self.transition_probas,
            ) = train(params, agent, self.rl_env, blockage_qn_net=queue_env)

            self.states_with_rewards[case_num] = self.get_reward()

            num_visit = self.states_with_visits.setdefault(case_num, 0) + 1
            self.states_with_visits[case_num] = num_visit

            self.num_states_explored += 1
            self.save_coverage_metric()
            self.save_key_states()
            self.save_peripheral_states()

            save_agent(agent)


if __name__ == "__main__":

    print(root_dir)

    # Filepath Used
    param_file = "user_config/eval_hyperparams.yml"
    config_file = "user_config/configuration.yml"

    # Parameters
    normal_std = 0.1

    # Initialize Engine
    params, hidden = load_hyperparams(param_file)

    rl_env = create_simulation_env(params, config_file)
    agent = create_ddpg_agent(rl_env, params, hidden)

    Engine = BreakdownEngine(rl_env, normal_std)
    Engine.run(agent)
