import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def plot_gradient(data_filepath, images_filepath, layer_name="layers.0.weight"):
    """
    Plots the gradient evolution of a specified layer in a neural network.

    Parameters:
    - data_filepath (str): The path to the directory containing 'gradient_dict.json'.
    - images_filepath (str): The path to the directory where the plot will be saved.
    - layer_name (str): The specific layer name to plot gradients for.

    The function saves the plot as 'Gradient.png' in the specified images directory.
    """
    plt.figure()
    filename = os.path.join(data_filepath, "gradient_dict.json")

    with open(filename, "r") as f:
        gradient_dict_loaded = json.load(f)

    layer_0_weight = gradient_dict_loaded[layer_name]
    num_params = len(gradient_dict_loaded[layer_name])

    param_evolution_all = []
    for i in range(num_params):
        param_evolution = [num_list[i] for num_list in layer_0_weight]
        param_evolution_all.append(param_evolution)

    for i, line in enumerate(param_evolution_all):
        plt.plot(line, label=f"Element {i + 1}")

    plt.title(f"Gradient of {layer_name}")
    plt.xlabel("List Index")
    plt.ylabel("Value")
    gradient_path = os.path.join(images_filepath, "Gradient Plots")
    os.makedirs(gradient_path, exist_ok=True)
    save_path = os.path.join(images_filepath, "Gradient Plots", "Gradient.png")
    plt.savefig(save_path)
    plt.close()


def plot_actor(data_filepath, images_filepath):
    """
    Plots the loss evolution of an actor model from training data.

    Parameters:
    - data_filepath (str): The path to the directory containing 'actor_loss.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Actor_loss.png' in the specified images directory.
    """
    pathname = os.path.join(data_filepath, "actor_loss.csv")
    actor_loss = pd.read_csv(pathname)

    plt.figure()
    plt.plot(actor_loss)
    plt.title("Actor Loss")
    loss_path = os.path.join(images_filepath, "Loss Plots")
    os.makedirs(loss_path, exist_ok=True)
    save_path = os.path.join(images_filepath, "Loss Plots", "Acotor_loss.png")
    plt.savefig(save_path)
    plt.close()


def plot_critic(data_filepath, images_filepath):
    """
    Plots the loss evolution of a critic model from training data.

    Parameters:
    - data_filepath (str): The path to the directory containing 'critic_loss.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Critic_loss.png' in the specified images directory.
    """
    pathname = os.path.join(data_filepath, "critic_loss.csv")
    critic_loss = pd.read_csv(pathname)

    plt.figure()
    plt.plot(critic_loss)
    plt.title("Critic Loss")

    save_path = os.path.join(images_filepath, "Loss Plots", "Critic_loss.png")
    plt.savefig(save_path)
    plt.close()


def plot_reward(data_filepath, images_filepath):
    """
    Plots the rewards obtained per episode and the average reward over all episodes.

    Parameters:
    - data_filepath (str): The path to the directory containing 'reward_dict.json'.
    - images_filepath (str): The path to the directory where the plots will be saved.

    The function saves the plots as 'Reward.png' and 'Average_reward.png' in the specified images directory.
    """
    plt.figure()
    filename = os.path.join(data_filepath, "reward_dict.json")

    with open(filename, "r") as f:
        reward_by_episodes = json.load(f)

    first_reward = []
    for key in reward_by_episodes.keys():
        first_reward.append(reward_by_episodes[key][0])

    plt.plot(first_reward)
    plt.title("First Reward Value/ Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    reward_path = os.path.join(images_filepath, "Reward Plots")
    os.makedirs(reward_path, exist_ok=True)
    save_path = os.path.join(images_filepath, "Reward Plots", "First_Reward.png")
    plt.savefig(save_path)

    plt.figure()
    plt.plot(reward_by_episodes[list(reward_by_episodes.keys())[-1]])
    plt.title("Reward Per Time step - Final Episode")
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    save_path = os.path.join(
        images_filepath, "Reward Plots", "Final_Learning_Curve.png"
    )
    plt.savefig(save_path)
    plt.close()


def plot_average_reward_episode(data_filepath, images_filepath):
    """
    Plots the average reward per episode over the course of training.

    Parameters:
    - data_filepath (str): The path to the directory containing 'reward_dict.json'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Reward_Avg_Per_Episode.png' in the specified images directory.
    """
    plt.figure()
    filename = os.path.join(data_filepath, "reward_dict.json")

    with open(filename, "r") as f:
        reward_data = json.load(f)

    reward_list = np.array(
        [reward_per_episode for reward_per_episode in reward_data.values()]
    )
    reward_average_per_episode = reward_list.mean(axis=1)
    plt.plot(reward_average_per_episode)
    plt.title("Average Reward/Episode")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    save_path = os.path.join(
        images_filepath, "Reward Plots", "Reward_Avg_Per_Episode.png"
    )
    plt.savefig(save_path)
    plt.close()


def plot_actor_vector(data_filepath, images_filepath):
    """
    Plots the evolution of action vectors produced by an actor model during training.

    Parameters:
    - data_filepath (str): The path to the directory containing 'action_dict.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Actor_space.png' in the specified images directory.
    """
    plt.figure()
    pathname = os.path.join(data_filepath, "action_dict.csv")
    action_vector = pd.read_csv(pathname)
    for column in action_vector.columns:
        plt.plot(action_vector.index, action_vector[column], label=column)

    plt.title("Action Vector")
    plt.legend(
        loc="upper left", bbox_to_anchor=(1, 1)
    )  # Coordinates for the legend box
    actor_space_path = os.path.join(images_filepath, "Actor_Space Plots")
    os.makedirs(actor_space_path, exist_ok=True)
    save_path = os.path.join(images_filepath, "Actor_Space Plots", "Actor_space.png")
    plt.savefig(save_path)
    plt.close()


def plot_transition_proba(
    data_filepath, images_filepath, transition_proba_dict, node=1
):
    """
    Plots the transition probabilities for a specified node based on training data.

    Parameters:
    - data_filepath (str): The path to the directory containing 'transition_proba.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.
    - transition_proba_dict (dict): A dictionary containing transition probabilities.
    - node (int): The node for which to plot the transition probabilities.

    The function saves the plot as 'transition_probas.png' in the specified images directory.
    """
    pathname = os.path.join(data_filepath, "transition_proba.csv")
    df = pd.read_csv(pathname)

    num_col = [int(index) for index in list(df.columns)]
    df.columns = num_col

    nodes = list(transition_proba_dict[node].keys())

    data_to_plot = df[nodes]
    plt.figure(figsize=(10, 6))
    for key, values in data_to_plot.items():
        plt.plot(values, label=f"Transitions from 1 to {key}")

    plt.title("Transition Probabilities from 1")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.legend()
    actor_space_path = os.path.join(images_filepath, "Transition_Proba Plots")
    os.makedirs(actor_space_path, exist_ok=True)
    save_path = os.path.join(
        images_filepath, "Transition_Proba Plots", "transition probas.png"
    )
    plt.savefig(save_path)
    plt.close()


def plot_reward_model_loss(data_filepath, images_filepath):
    """
    Plots the loss of the reward model during training.

    Parameters:
    - data_filepath (str): The path to the directory containing 'reward_model_loss.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Reward_model_loss.png' in the specified images directory.
    """
    pathname = os.path.join(data_filepath, "reward_model_loss.csv")
    reward_model_loss = pd.read_csv(pathname, index_col=0)

    plt.figure()
    plt.plot(reward_model_loss.rolling(window=12).mean())
    plt.title("reward_model_loss")

    save_path = os.path.join(images_filepath, "Loss Plots", "Reward_model_loss.png")
    plt.savefig(save_path)
    plt.close()


def plot_next_state_model_loss(data_filepath, images_filepath):
    """
    Plots the loss of the next state prediction model during training.

    Parameters:
    - data_filepath (str): The path to the directory containing 'next_state_model_loss.csv'.
    - images_filepath (str): The path to the directory where the plot will be saved.

    The function saves the plot as 'Next_model_loss.png' in the specified images directory.
    """
    pathname = os.path.join(data_filepath, "next_state_model_loss.csv")
    next_state_model = pd.read_csv(pathname)

    plt.figure()
    plt.plot(next_state_model)
    plt.title("next_state_model_loss")

    save_path = os.path.join(images_filepath, "Loss Plots", "Next_model_loss.png")
    plt.savefig(save_path)
    plt.close()


def plot(data_filepath, images_filepath, transition_probas=None, filepath=None):
    """
    Central function to execute all individual plot functions for model training evaluation.

    Parameters:
    - data_filepath (str): The base path to the directory containing all required data files.
    - images_filepath (str): The base path to the directory where all plots will be saved.
    - transition_probas (dict, optional): Dictionary containing transition probabilities, required for plotting transition probabilities.

    This function calls all individual plotting functions and saves their outputs in the specified directory.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if filepath is None:
        filepath = os.path.join(base_path, "foundations", "output_plots")
    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    plot_reward(data_filepath, images_filepath)
    plot_average_reward_episode(data_filepath, images_filepath)
    plot_actor(data_filepath, images_filepath)
    plot_critic(data_filepath, images_filepath)
    plot_actor_vector(data_filepath, images_filepath)
    plot_gradient(data_filepath, images_filepath)
    plot_transition_proba(data_filepath, images_filepath, transition_probas, node=1)
    plot_reward_model_loss(data_filepath, images_filepath)
    plot_next_state_model_loss(data_filepath, images_filepath)

    print(f"plots have been saved at {filepath}")
