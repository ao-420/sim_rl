# This script is used to automatically train the agent for varying numbers of training episodes
# and then evaluate the performance of each agent on the
# simulation environment - using total reward over time as the metric for evaluation.

# Change this so that it mimics the gradient based approach


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


class ConvergenceEvaluation(Engine):
    def __init__(
        self, window_size, threshold, consecutive_points, timesteps=100, num_sim=100
    ):
        """
        This class is responsible for tracking the changes in the reward per episode as the agent is trained
        and determining the point at which the agent's performance stabilizes.

        Parameters:
            window_size (int): Number of points used to compute the moving average of rewards.
            threshold (float): Threshold for the derivative of rewards to determine stabilization.
            consecutive_points (int): Number of consecutive points below the threshold to declare convergence.
            timesteps (int): Number of timesteps per simulation. Defaults to 100.
            num_sim (int): Number of simulations to average for evaluation. Defaults to 100.
        """
        self.timesteps = timesteps
        self.total_rewards = []
        self.num_sim = num_sim
        self.window_size = window_size
        self.threshold = threshold
        self.consecutive_points = consecutive_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reward_data_from_json = self.load_json_data()
        self.commence_analysis = (
            10  # The variable controls when the analysis should commence
        )
        self.eval_interval = 10  # Interval for evaluating the agent during training
        self.episode_rewards = {"episodes": [], "rewards": []}

    def load_json_data(self):
        # Current directory
        current_dir = os.path.dirname(os.path.dirname(os.getcwd()))

        # Construct the relative path to the target JSON file
        relative_path = os.path.join(
            root_dir, "foundations", "output_csv", "reward_dict.json"
        )

        # Normalize the path to avoid any cross-platform issues
        normalized_path = os.path.normpath(relative_path)

        # Load the JSON file using a context manager
        with open(normalized_path, "r") as file:
            data = json.load(file)
        return data

    def moving_average(self, data):
        """Calculate moving average of the data using the defined window size."""
        return np.convolve(
            data, np.ones(self.window_size) / self.window_size, mode="valid"
        )

    def calculate_derivative(self, data):
        """Calculate the first derivative of the data."""
        return np.diff(data)

    def find_stabilization_point(self, derivatives):
        abs_derivatives = np.abs(derivatives)
        count = 0  # Counter for consecutive points under threshold
        for i in range(len(abs_derivatives)):
            if abs_derivatives[i] < self.threshold:
                count += 1
                if count >= self.consecutive_points:
                    return (
                        i - self.consecutive_points + 2
                    )  # Adjust for the window of points
            else:
                count = 0  # Reset counter if the point is above the threshold
        return -1  # Returns -1 if no stabilization point is found

    def compute_episode_rewards(data_dict):
        """This function calculates the sum of the the rewards for each episode in the dictionary.

        Parameters:
            data_dict (dictionary): dictionary of lists of rewards for each episode

        Returns:
            list: list of the sum of rewards for each episode
        """
        sums_list = []
        for values in data_dict.values():
            total_sum = sum(values)
            sums_list.append(total_sum)
        return sums_list

    def evaluate_convergence(self, reward_data=None):
        """Evaluate the startup behavior of the agent."""
        if reward_data is None:
            reward_data = self.reward_data_from_json

        self.rewards = self.compute_episode_rewards(reward_data)
        self.smoothed_rewards = self.moving_average(self.rewards)
        derivatives = self.calculate_derivative(self.smoothed_rewards)
        self.stabilization_point = self.find_stabilization_point(derivatives)
        # self.plot_results()
        return self.stabilization_point

    def train(self, params, agent, env, best_params=None, blockage_qn_net=None):
        """
        Modified training function to include the logic for stopping the training process when the stabilization point is reached.

        Parameters:
        - params (dict): Hyperparameters for training.
        - agent: The agent to be trained.
        - env: The environment in which the agent operates.

        Returns:
        - Multiple values including lists that track various metrics through training.
        """
        self.save_agent(agent)
        if best_params is not None:
            for key in params.keys():
                if key not in best_params.keys():
                    best_params[key] = params[key]
            params = best_params

        next_state_model_list_all = []
        reward_model_list_all = []
        gradient_dict_all = {}
        action_dict = {}
        transition_probas = self.init_transition_proba(env)
        actor_loss_list = []
        critic_loss_list = []
        reward_by_episode = {}
        num_episodes, _, num_epochs, time_steps, _, num_train_AC = (
            self.get_params_for_train(params)
        )
        latest_transition_proba = None

        for episode in tqdm(range(num_episodes), desc="Episode Progress"):
            agent.train()
            if blockage_qn_net is None:
                env.reset()
            else:
                env.reset(blockage_qn_net)

            if latest_transition_proba is not None:
                env.net.set_transitions(latest_transition_proba)

            env.simulate()
            update = 0
            reward_list = []

            for _ in tqdm(range(time_steps), desc="Time Steps Progress"):
                state = env.get_state()
                state_tensor = torch.tensor(state)
                action = agent.select_action(state_tensor).to(self.device)
                action_list = action.cpu().numpy().tolist()

                for index, value in enumerate(action_list):
                    node_list = action_dict.setdefault(index, [])
                    node_list.append(value)
                    action_dict[index] = node_list

                next_state_tensor = (
                    torch.tensor(env.get_next_state(action)).float().to(self.device)
                )
                reward = env.get_reward()
                reward_list.append(reward)
                experience = (state_tensor, action, reward, next_state_tensor)
                agent.store_experience(experience)

            reward_model_loss_list, next_state_loss_list = agent.fit_model(
                batch_size=time_steps, epochs=num_epochs
            )
            next_state_model_list_all += next_state_loss_list
            reward_model_list_all += reward_model_loss_list
            transition_probas = self.update_transition_probas(transition_probas, env)

            for _ in tqdm(range(num_train_AC), desc="Train Agent"):
                batch = agent.buffer.sample(batch_size=time_steps)
                critic_loss = agent.update_critic_network(batch)
                actor_loss, gradient_dict = agent.update_actor_network(batch)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

            agent.plan(batch)
            agent.soft_update(network="critic")
            agent.soft_update(network="actor")
            gradient_dict_all[update] = gradient_dict
            agent.buffer.clear()
            reward_by_episode[episode] = reward_list
            latest_transition_proba = env.transition_proba

            # Logic for handling periodic evaluation of the agent
            if episode % self.eval_interval == 0:
                episode_reward = self.start_evaluation(
                    env, agent, time_steps, num_simulations=self.num_sim
                )
                self.episode_rewards["episodes"].append(episode)
                self.episode_rewards["rewards"].append(episode_reward)

            # Logic for handling the stabilization evaluation
            if episode > self.commence_analysis:
                stabilization_point = self.evaluate_convergence(
                    self.episode_rewards["rewards"]
                )  # Evalaute whether the stabilization point has been reached
                if stabilization_point != -1:
                    print(f"Stabilization point found at episode {episode}")
                    break  # Exit the training loop as stabilization point is found

        self.save_agent(agent)
        # Return all the collected data
        return (
            next_state_model_list_all,
            critic_loss_list,
            actor_loss_list,
            reward_by_episode,
            action_dict,
            gradient_dict,
            transition_probas,
        )

    def plot_results(self):
        """Plot the original and smoothed rewards with stabilization point."""
        plt.figure(figsize=(10, 5))
        episodes = self.episode_rewards["episodes"]
        if len(episodes) != len(self.rewards):
            raise ValueError(
                "The length of episodes list and rewards list must be the same."
            )

        plt.plot(episodes, self.rewards, label="Original Rewards", alpha=0.5)
        plt.plot(episodes, self.smoothed_rewards, label="Smoothed Rewards", color="red")

        # Plot the stabilization point if it exists
        if self.stabilization_point != -1:
            # Find the episode corresponding to the stabilization point index
            if self.stabilization_point < len(episodes):
                stabilization_episode = episodes[self.stabilization_point]
                plt.axvline(
                    x=stabilization_episode, color="green", label="Stabilization Point"
                )
            else:
                print("Stabilization point is out of the episode range.")

        plt.title("Reward Stabilization Analysis - Cutoff Episode " + str(self.episode))
        plt.xlabel("Episodes")
        plt.ylabel("Reward per Episode")
        plt.legend()
        plt.savefig("reward_plot.png", dpi=1000)
        plt.close()
        print("Plot saved as 'reward_plot.png'.")


if __name__ == "__main__":

    # Logic for using this class
    # Define the parameters for the startup behavior analysis
    window_size = 5
    threshold = 0.01
    consecutive_points = 5

    # Create the startup behavior analysis engine and evaluate the stabilization point
    convergence_eval = ConvergenceEvaluation(window_size, threshold, consecutive_points)

    # 2. Specify the file path to the agent's configuration yaml file
    agent = "user_config/eval_hyperparams.yml"

    # 3. Speficy the file path for the training and evaluation environment's configuration yaml file
    env = "user_config/configuration.yml"

    # 4. Initialize the training and evaluation process
    convergence_eval.start_train(
        env,
        agent,
        save_file=True,
        data_filename="output_csv",
        image_filename="output_plots",
    )
