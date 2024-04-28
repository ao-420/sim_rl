import sys
import os
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
parent_dir = os.path.dirname(os.path.dirname(root_dir))
os.chdir(parent_dir)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import torch
import matplotlib.pyplot as plt
from foundations.core_functions import *
from foundations.core_functions import create_simulation_env
from foundations.core_functions import Engine
from rl_env.RL_Environment import *
from queue_env.queueing_network import *
from queueing_tool.queues.queue_servers import *
import numpy as np
import copy
import os
import yaml


class ControlEvaluation(Engine):

    def __init__(self, queue_index=2, metric="throughput"):
        """
        This class reponsible for evaluating the learnt policy of the agent in a simulated environment.
        Parameters:
        queue_index (int): Index of the queue for which metrics are evaluated.
        metric (str): The performance metric to evaluate (Available options: 'throughput', 'waiting_time').
        """

        self.config_param_filepath = "user_config/configuration.yml"
        self.eval_param_filepath = "user_config/eval_hyperparams.yml"
        params, hidden = self.load_hyperparams(self.eval_param_filepath)
        self.environment = create_simulation_env(
            params, self.config_param_filepath, disrupt_case=False
        )

        self.agent = create_ddpg_agent(self.environment, params, hidden)
        self.metric = metric
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.queue_index = queue_index
        self.queue_metrics = []
        self.case = "Normal"

    def load_hyperparams(self, eval_param_filepath):
        """
        Load hyperparameters from a YAML file.

        Parameters:
        - param_filepath (str): The file path to the hyperparameters YAML file.

        Returns:
        - tuple: A tuple containing two dictionaries, `params` for hyperparameters and `hidden` for hidden layer configurations.
        """

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_dir = os.path.dirname(script_dir)
        abs_file_path = os.path.join(project_dir, eval_param_filepath)

        with open(abs_file_path, "r") as env_param_file:
            parameter_dictionary = yaml.load(env_param_file, Loader=yaml.FullLoader)

        params = parameter_dictionary["rl_params"]
        hidden = parameter_dictionary["network_params"]

        return params, hidden

    def plot_transition_proba(self, transition_proba_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels."""
        self.ax.clear()  # Clear previous plots
        self.ax.plot(transition_proba_lists)
        self.ax.set(
            xlabel="Time Steps",
            ylabel="Transition Probability",
            title=f"Transition Probability vs Time Steps for Queue {self.queue_index}",
        )
        figure_name = f"{self.case}_transition_proba_plot.png"

        script_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        evaluation_dir = "evaluation/decision_evaluation/output_plots"

        full_path = os.path.join(script_dir, evaluation_dir)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        save_filepath = os.path.join(full_path, figure_name)

        plt.savefig(save_filepath)

    def plot_queue(self, labels, *queue_metrics_lists):
        """Plotting function that supports a variable number of queue metrics lists and labels."""
        self.fig, self.ax = plt.subplots()
        self.ax.clear()  # Clear previous plots
        for queue_metrics, label in zip(queue_metrics_lists, labels):
            self.ax.plot(range(len(queue_metrics)), queue_metrics, label=label)
        self.ax.set(
            xlabel="Time Steps",
            ylabel=self.metric,
            title=self.metric + f" vs Time Steps for Queue {self.queue_index}",
        )
        figure_name = f"{self.case}_{self.metric}_plot.png"

        script_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        evaluation_dir = "evaluation/decision_evaluation/output_plots"

        full_path = os.path.join(script_dir, evaluation_dir)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        save_filepath = os.path.join(full_path, figure_name)

        plt.savefig(save_filepath)

    def start_evaluation(self, agent, time_steps):
        """Modified evaluation function that supports plotting of queue metrics and transition probabilities."""
        source_edge = self.environment.net.edge2queue[self.queue_index].edge[0]
        target_edge = self.environment.net.edge2queue[self.queue_index].edge[1]

        queue_metrics = []
        queue_transition_proba = []

        self.environment.simulate()
        for _ in range(time_steps):
            state = self.environment.get_state()
            action = agent.actor(state).detach()
            state = self.environment.get_next_state(action)[0]
            queue_metrics.append(
                self.environment.return_queue(self.queue_index, metric=self.metric)
            )
            queue_transition_proba.append(
                self.environment.transition_proba[source_edge][target_edge]
            )
        self.plot_queue([self.metric], queue_metrics)
        self.plot_transition_proba(queue_transition_proba)

        return queue_metrics, queue_transition_proba

    def load_agent(self):
        project_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        agent_path = os.path.join(project_dir, "agents", "trained_agent.pt")
        agent = torch.load(agent_path)
        return agent


class DisruptionEvaluation(ControlEvaluation):
    """
    This class extends Control Evaluation for demonstrating scenarios where the agent has to handle disruptions.
    """

    def __init__(self, queue_index, metric):
        super().__init__(queue_index, metric)

        self.config_param_filepath = "user_config/configuration.yml"
        self.eval_param_filepath = "user_config/eval_hyperparams.yml"
        params, hidden = self.load_hyperparams(self.eval_param_filepath)
        self.queue_index = queue_index
        self.metric = metric
        self.case = "Blocked"
        self.standard_environment = create_simulation_env(
            params, self.config_param_filepath, disrupt_case=False
        )
        self.deactivate_node = self.standard_environment.net.edge2queue[
            self.queue_index
        ].edge[1]
        self.environment = create_simulation_env(
            params,
            self.config_param_filepath,
            disrupt_case=True,
            disrupt=True,
            queue_index=self.queue_index,
            deactivate_node=self.deactivate_node,
        )

    def plot_transition_proba_changes(
        self,
        queue_transition_proba_before_disrupt,
        queue_transition_proba_after_disrupt,
    ):
        self.param_file = (
            "user_config\\evaluation_params\\blockage_demonstration_params.yml"
        )
        self.save_file = "evaluation\\decision_evaluation\output_plots"
        trans_proba_changes = (
            queue_transition_proba_before_disrupt + queue_transition_proba_after_disrupt
        )
        plt.figure()
        plt.plot(trans_proba_changes)
        plt.title(
            f"Routing Proba Changes Before/After Disruption for Queue_{queue_index}"
        )

        script_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        evaluation_dir = "evaluation/decision_evaluation/output_plots"
        figure_name = "Transition_Proba_BeforeAfter_Blockage.png"

        full_path = os.path.join(script_dir, evaluation_dir)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        save_filepath = os.path.join(full_path, figure_name)

        plt.savefig(save_filepath)


# Example Usage
if __name__ == "__main__":

    # Define the parameters for initializing the CoNtrol Envaluation Object
    queue_index = 2
    metric = "throughput"
    nc = ControlEvaluation(queue_index=queue_index, metric=metric)

    # Define the standard parameters for starting the evaluation
    time_steps = 100
    sim_jobs = 100
    env = "user_config/configuration.yml"
    agent = nc.load_agent()
    _, queue_transition_proba_before_disrupt = nc.start_evaluation(
        agent=agent, time_steps=time_steps
    )

    ## Static Disruption
    queue_index = 2
    metric = "throughput"
    sd = DisruptionEvaluation(queue_index, metric)
    _, queue_transition_proba_after_disrupt = sd.start_evaluation(
        agent=agent, time_steps=time_steps
    )

    # Changes in transition proba
    # _, queue_transition_proba_before_disrupt = nc.evaluation(agent=agent, time_steps=time_steps)
    # _, queue_transition_proba_after_disrupt = sd.evaluation(agent= agent, time_steps=time_steps)
    sd.plot_transition_proba_changes(
        queue_transition_proba_before_disrupt, queue_transition_proba_after_disrupt
    )