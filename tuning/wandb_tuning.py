import os
from tqdm import tqdm

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
parent_dir = os.path.dirname(os.path.dirname(root_dir))
os.chdir(parent_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from agents.ddpg_agent import DDPGAgent
from rl_env.RL_Environment import RLEnv
import torch
import numpy as np
import wandb
import yaml
import os
from queue_env.queue_base_functions import *

def load_hyperparams(eval_param_filepath):
    """
    Load hyperparameters from a YAML file.

    Parameters:
    - param_filepath (str): The file path to the hyperparameters YAML file.

    Returns:
    - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    abs_file_path = os.path.join(project_dir, eval_param_filepath)

    with open(abs_file_path, "r") as env_param_file:
        parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)
    params = parameter_dictionary["rl_params"]
    hidden = parameter_dictionary["network_params"]

    return params, hidden


def load_tuning_config(tune_param_filepath):

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    abs_file_path = os.path.join(project_dir, tune_param_filepath)

    with open(abs_file_path, "r") as tune_params_file:
        tune_params = yaml.load(tune_params_file, Loader=yaml.FullLoader)

    config = {
        "method": "bayes",  # or 'grid', 'random'
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "min": tune_params["learning_rate_min"],
                "max": tune_params["learning_rate_max"],
            },
            "epochs": {"values": tune_params["epochs_list"]},
            "batch_size": {"values": tune_params["batch_size"]},
            "tau": {"min": tune_params["tau_min"], "max": tune_params["tau_max"]},
            "discount": {
                "min": tune_params["discount_min"],
                "max": tune_params["discount_max"],
            },
            "epsilon": {
                "min": tune_params["epsilon_min"],
                "max": tune_params["epsilon_max"],
            },
            "planning_steps": {"values": tune_params["planning_steps"]},
            "num_sample": {"values": tune_params["num_sample"]},
            "w1": {"values": tune_params["w1"]},
            "w2": {"values": tune_params["w2"]},
            "epsilon_state_exploration": {
                "values": tune_params["epsilon_state_exploration"]
            },
            "num_episodes": {"values": tune_params["num_episodes"]},
            "target_update_frequency": {
                "values": tune_params["target_update_frequency"]
            },
            "time_steps": {"values": tune_params["time_steps"]},
        },
    }

    return config


def get_agent_parameters(config):
    params = {}
    params["tau"] = config["tau"]
    params["learning_rate"] = config["learning_rate"]
    params["discount"] = config["discount"]
    params["epsilon"] = config["epsilon"]
    params["planning_steps"] = config["planning_steps"]

    return params


def init_wandb(
    project_name,
    tune_param_filepath,
    config_param_filepath,
    eval_param_filepath,
    opt_target="reward",
    num_runs=100,
):
    # initialize W&B
    wandb.login()
    wandb.init(project=project_name)

    # read hyperparameter files
    tuning_config = load_tuning_config(tune_param_filepath)
    env_config = load_config(config_param_filepath)
    _, hidden = load_hyperparams(eval_param_filepath)
    sweep_id = wandb.sweep(tuning_config, project=project_name)

    def wandb_train():
        with wandb.init() as run:
            queue_env = create_queueing_env(config_param_filepath)
            env = create_RL_env(queue_env, env_config)

            config = run.config
            num_sample = config.num_sample  # Access 'num_sample' directly
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            w1 = config.w1  # Access 'w1' directly
            w2 = config.w2  # Access 'w2' directly
            epsilon_state_exploration = (
                config.epsilon_state_exploration
            )  # Direct access
            num_episodes = config.num_episodes  # Direct access
            batch_size = config.batch_size  # Direct access
            num_epochs = config.epochs  # Access 'epochs' directly
            time_steps = config.time_steps  # Direct access
            target_update_frequency = config.target_update_frequency  # Direct access

            reward_list = []
            action_dict = {}

            n_states = len(env.get_state())
            n_actions = len(env.get_state()) - 2  # need to modify

            agent_parameters = get_agent_parameters(config)
            agent = DDPGAgent(
                n_states, n_actions, hidden=hidden, params=agent_parameters
            ).to(device)

            agent.train()
            for episode in tqdm(range(num_episodes), desc="Training Progress"):
                env.reset()
                state = env.explore_state(agent, env.qn_net, episode)
                t = 0
                while t < time_steps:

                    if type(state) == np.ndarray:
                        state = torch.from_numpy(state).to(device)
                    action = agent.select_action(state).to(device)

                    action_list = action.cpu().numpy().tolist()
                    for index, value in enumerate(action_list):
                        node_list = action_dict.setdefault(index, [])
                        node_list.append(value)
                        action_dict[index] = node_list

                    next_state = env.get_next_state(action)
                    next_state = torch.tensor(next_state).float().to(device)
                    reward = env.get_reward()

                    reward_list.append(reward)
                    experience = (state, action, reward, next_state)
                    agent.store_experience(experience)

                    if agent.buffer.current_size > batch_size:

                        agent.fit_model(batch_size=batch_size, epochs=num_epochs)

                        batch = agent.buffer.sample(batch_size=batch_size)
                        agent.update_critic_network(batch)
                        agent.update_actor_network(batch)
                        agent.plan(batch)

                    t += 1
                    state = next_state

                    if t % target_update_frequency == 0:
                        agent.soft_update(network="critic")
                        agent.soft_update(network="actor")
                wandb.log({"episode": episode, opt_target: np.mean(reward_list)})

    wandb.agent(sweep_id, wandb_train, count=num_runs)


def get_best_param(project_name, opt_target="reward"):
    api = wandb.Api()
    sweep = api.sweep(project_name)
    runs = sorted(
        sweep.runs, key=lambda r: r.summary.get(opt_target, float("inf")), reverse=True
    )  # Set reverse=False for metrics where lower is better

    best_run = runs[0]

    print("Best Hyperparameters:")
    for key, value in best_run.config.items():
        print(f"{key}: {value}")

    best_hyperparameters = best_run.config
    return best_hyperparameters


if __name__ == "__main__":
    project_name = "datasparq"
    num_runs = 10
    tune_param_filepath = "user_config/tuning_hyperparams.yml"
    config_param_filepath = "user_config/configuration.yml"
    eval_param_filepath = "user_config/eval_hyperparams.yml"
    init_wandb(
        project_name,
        tune_param_filepath,
        config_param_filepath,
        eval_param_filepath,
        num_runs=num_runs,
        opt_target="reward",
    )
