import os
from tqdm import tqdm

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.insert(0,str(root_dir))

from agents.ddpg_agent import DDPGAgent
from rl_env.RL_Environment import RLEnv
import torch
import numpy as np
import yaml
import os
import random
from foundations.core_functions import *
from queue_env.queue_base_functions import *

import ray
from ray import train as train_ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

global rewards_list
rewards_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_gpu():
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "This operation requires a GPU, but none is available. Please run on GPU"
        )


def load_tuning_config(tune_param_filepath):

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory to the MScDataSparqProject directory
    project_dir = os.path.dirname(script_dir)

    # Build the path to the configuration file
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
            "time_steps": {"values": tune_params["time_steps"]},
            "target_update_frequency":{"values": tune_params["target_update_frequency"]},
            "num_train_AC":{"values": tune_params["num_train_AC"]}
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
    params["critic_lr"] = config["learning_rate"]
    params["actor_lr"] = config["learning_rate"]
    params["planning_std"] = config["epsilon"]
    params['buffer_size'] = 10000

    return params


def train(config, eval_param_filepath="user_config/eval_hyperparams.yml"):

    environment = create_simulation_env(
        params={"num_sim": 5000, "temperature": 0.15}, config_file="user_config/configuration.yml"
    )

    n_states = len(environment.get_state())
    n_actions = environment.net.num_nodes - environment.num_nullnodes

    agent_params = get_agent_parameters(config)
    _, hidden = load_hyperparams(eval_param_filepath)

    agent = DDPGAgent(n_states, n_actions, hidden=hidden, params=agent_params,device=device)
    env = environment

    print(config)
    num_episodes = config["num_episodes"]
    time_steps = config["time_steps"]
    threshold = config["threshold"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    target_update_frequency = config["target_update_frequency"]
    num_train_AC = config['num_train_AC']

    transition_probas = init_transition_proba(env)
    reward_list = []
    action_dict = {}

    latest_transition_proba = None

    for episode in tqdm(range(num_episodes), desc="Episode Progress"):
        agent.train()

        if latest_transition_proba is not None:
            env.net.set_transitions(latest_transition_proba)

        env.simulate()
        update = 0

        for _ in tqdm(range(batch_size), desc="Time Steps Progress"):

            state = env.get_state()

            state_tensor = torch.tensor(state)
            action = agent.select_action(state_tensor).to(device)
            action_list = action.cpu().numpy().tolist()

            for index, value in enumerate(action_list):
                node_list = action_dict.setdefault(index, [])
                node_list.append(value)
                action_dict[index] = node_list

            next_state_tensor = (
                torch.tensor(env.get_next_state(action)).float().to(device)
            )
            reward = env.get_reward()
            if np.isnan(reward):
                continue
            else:
                reward_list.append(reward)
                ray.train.report({"reward": reward})
            experience = (state_tensor, action, reward, next_state_tensor)
            agent.store_experience(experience)

        agent.fit_model(batch_size=batch_size, epochs=epochs)
        transition_probas = update_transition_probas(transition_probas, env)

        for _ in tqdm(range(num_train_AC), desc="Train Agent"):

            batch = agent.buffer.sample(batch_size=batch_size)
            agent.update_critic_network(batch)
            agent.update_actor_network(batch)

        agent.plan(batch)
        agent.soft_update(network="critic")
        agent.soft_update(network="actor")
        agent.buffer.clear()
        latest_transition_proba = env.transition_proba
    return {"reward": np.mean(np.array(reward_list))}


def ray_tune():

    require_gpu()

    config = load_tuning_config(
        tune_param_filepath="user_config/tuning_hyperparams.yml"
    )

    hyperparam_mutations = {
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "train_batch_size": lambda: random.randint(8, 64),
        "num_episodes": lambda: random.randint(10, 50),
        "max_episode_length": lambda: random.randint(8, 64),
        "batch_size_buffer_sampling": lambda: random.randint(8, 64),
        "batch_size": lambda: random.randint(8, 64),
        "epochs": lambda: random.randint(8, 64),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=train,
    )

    param_space = {
        "num_workers": 10,
        "num_cpus": 0,  # number of CPUs to use per trial
        "num_gpus": 1,  # number of GPUs to use per trial
        # These params are tuned from a fixed starting value.
        "learning_rate": 1e-4,
        # These params start off randomly drawn from a set.
        "tau": tune.uniform(
            config["parameters"]["tau"]["min"], config["parameters"]["tau"]["max"]
        ),
        "discount": tune.uniform(
            config["parameters"]["discount"]["min"],
            config["parameters"]["discount"]["max"],
        ),
        "epsilon": tune.uniform(
            config["parameters"]["epsilon"]["min"],
            config["parameters"]["epsilon"]["max"],
        ),
        "planning_steps": tune.choice(config["parameters"]["planning_steps"]["values"]),
        "num_episodes": tune.choice(config["parameters"]["num_episodes"]["values"]),
        "threshold": tune.choice(config["parameters"]["batch_size"]["values"]),
        "time_steps": tune.choice(config["parameters"]["time_steps"]["values"]),
        "batch_size": tune.choice(config["parameters"]["batch_size"]["values"]),
        "epochs": tune.choice(config["parameters"]["epochs"]["values"]),
        "target_update_frequency": tune.choice(config["parameters"]["target_update_frequency"]["values"]),
        "num_train_AC": tune.choice(config["parameters"]["num_train_AC"]["values"])
    }
    tuner = tune.Tuner(
        tune.with_resources(train, resources={"gpu": 0.5, "cpu": 0.5}),
        tune_config=tune.TuneConfig(
            metric="reward",
            mode="max",
            num_samples=10,
            scheduler=pbt,
        ),
        param_space=param_space,
        run_config=train_ray.RunConfig(),
    )

    ray.init(
    runtime_env={
        "working_dir": root_dir,
    })


    # Start Tuning
    results = tuner.fit()

    return results


if __name__ == "__main__":
    print(root_dir)
    ray_tune()
