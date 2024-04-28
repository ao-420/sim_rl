import sys
import os
from pathlib import Path

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
parent_dir = os.path.dirname(os.path.dirname(root_dir))
os.chdir(parent_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from foundations.core_functions import Engine


class StartupBehavior(Engine):
    def __init__(self, window_size, threshold, consecutive_points, episode):
        """
        This class is responsible for tracking the changes in the reward per timestep within an epsiode as the agent is trained
        and determining the point at which the agent's performance stabilizes.
        Parameters:
        window_size (int): The number of episodes over which the moving average is computed.
        threshold (float): The threshold for the derivative of the moving average, below which
                           the system is considered to have stabilized.
        consecutive_points (int): The required number of consecutive points that must meet the
                                  threshold criterion to declare stabilization.
        episode (int): Episode number to specifically analyze for stabilization.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.consecutive_points = consecutive_points
        self.episode = episode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.commence_analysis = (
            10  # The variable controls when the analysis should commence
        )
        self.data_path = self.get_data_path()

    def get_data_path(self):
        # Current directory
        current_dir = root_dir.parent.parent

        # Construct the relative path to the target JSON file
        relative_path = os.path.join(
            current_dir, "foundations", "output_csv", "reward_dict.json"
        )

        # Normalize the path to avoid any cross-platform issues
        normalized_path = os.path.normpath(relative_path)

        return normalized_path

    def load_json_data(self, datapath):

        # Load the JSON file using a context manager
        with open(datapath, "r") as file:
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

    def evaluate_convergence(self, reward_data=None):
        """Evaluate the startup behavior of the agent."""

        datapath = self.get_data_path()
        reward_data_from_json = self.load_json_data(datapath)

        if reward_data is None:
            reward_data = reward_data_from_json

        self.rewards = reward_data[str(self.episode)]
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

            # Additonal logic for handling the stabilization evaluation
            if episode > self.commence_analysis:
                self.episode = episode
                stabilization_point = self.evaluate_convergence(
                    reward_by_episode
                )  # Evalaute whether the stabilization point has been reached
                if stabilization_point != -1:
                    print(
                        f"Stabilization point found at episode: {episode} and timestep: {stabilization_point}"
                    )
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
        plt.plot(self.rewards, label="Original Rewards", alpha=0.5)
        plt.plot(
            range(len(self.smoothed_rewards)),
            self.smoothed_rewards,
            label="Smoothed Rewards",
            color="red",
        )
        if self.stabilization_point != -1:
            plt.axvline(
                x=self.stabilization_point, color="green", label="Stabilization Point"
            )
        plt.title("Burn in Period Analysis - Episode " + str(self.episode))
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("reward_plot.png")
        plt.close()
        print("Plot saved as 'reward_plot.png'.")

    def print_results(self):
        if self.stabilization_point == -1:
            print(f"No Stabilization point found")
        else:
            print(f"Stabilization point found at timestep: {self.stabilization_point}")


if __name__ == "__main__":

    # Define the parameters for the startup behavior analysis
    window_size = 5
    threshold = 0.01
    consecutive_points = 5
    episode = 0  # The episode to evaluate for a saved json file

    # Create the startup behavior analysis engine and evaluate the stabilization point
    startup_evaluation = StartupBehavior(
        window_size, threshold, consecutive_points, episode
    )

    # Implement the original evaluation process on saved data from a json file
    stabilization_point = startup_evaluation.evaluate_convergence()

    # Automatically run the training process and stop when the stabilization point is reached
    agent = "user_config/eval_hyperparams.yml"
    env = "user_config/configuration.yml"
    startup_evaluation.start_train(
        env,
        agent,
        save_file=True,
        data_filename="output_csv",
        image_filename="output_plots",
    )

    # Plot the results and evaluate the attributes for addtional analysis
    startup_evaluation.plot_results()
    startup_evaluation.print_results()
