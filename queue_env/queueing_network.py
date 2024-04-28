import numpy as np
import yaml
import os
import sys

# Append the path where the queueing_tool package is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
queue_foundations_dir = os.path.join(project_dir, "queue_env", "queue_foundations")
sys.path.append(queue_foundations_dir)

from queueing_tool.network.queue_network import QueueNetwork
from queueing_tool.graph.graph_wrapper import adjacency2graph
from queueing_tool.queues.agents import Agent
from queueing_tool.queues.queue_servers import *


class Queue_network:
    def __init__(self):
        """
        Initializes a new instance of the Queue_network class.
        """
        pass

    def process_config(self, filename):
        """
        This function accepts the name of the yaml file as the input and returns the variables for the process_input function.

        Parameters:
        - filename (str) : Name of the yaml file

        Returns:
        - lambda_list (list) : List of arrival rates for each queue
        - miu_list (list) : List of service rates for each queue
        - active_cap (int) : Active capacity of the server
        - deactive_t (float) : Deactivation time
        - adjacent_list (dict) : Adjacency list of the network
        - buffer_size_for_each_queue (list) : List of buffer sizes for each queue
        - transition_proba (dict) : Transition probability matrix
        """
        parameters = open(filename, "r")
        parameter_dictionary = yaml.load(parameters, Loader=yaml.FullLoader)
        lambda_list = parameter_dictionary["lambda_list"]
        lambda_list = [float(i) for i in lambda_list]
        miu_list = parameter_dictionary["miu_list"]
        miu_list = [float(i) for i in miu_list]
        active_cap = parameter_dictionary["active_cap"]
        active_cap = float(active_cap)
        deactive_cap = parameter_dictionary["deactive_cap"]
        deactive_cap = float(deactive_cap)
        adjacent_list = parameter_dictionary["adjacent_list"]
        adjacent_list = {int(k): [int(i) for i in v] for k, v in adjacent_list.items()}
        buffer_size_for_each_queue = parameter_dictionary["buffer_size"]
        buffer_size_for_each_queue = [int(i) for i in buffer_size_for_each_queue]
        if "transition_proba" in parameter_dictionary.keys():
            transition_proba = parameter_dictionary["transition_proba"]
        else:
            transition_proba = None
        return (
            lambda_list,
            miu_list,
            active_cap,
            deactive_cap,
            adjacent_list,
            buffer_size_for_each_queue,
            transition_proba,
        )

    def process_input(
        self,
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba,
        max_agents,
        sim_jobs,
    ):
        """
        Configures the queue network simulation environment with provided inputs.

        Parameters:
        - arrival_rate (float): The overall rate at which jobs arrive at the queue network.
        - miu_list (list): List of service rates for each queue.
        - q_classes (dict): Mapping of queue identifiers to their respective queue class types.
        - q_args (dict): Additional arguments specific to each queue class.
        - adjacent_list (dict): Adjacency list representing the connections between queues.
        - edge_list (dict): Detailed edge list providing specific connections and identifiers.
        - transition_proba (dict): Probabilities of transitioning from one queue to another.
        - max_agents (int): Maximum number of concurrent agents in the network.
        - sim_jobs (int): Total number of jobs to simulate.
        """
        # param for first server
        self.lamda = arrival_rate
        self.miu = miu_list

        # Configure the network
        self.adja_list = adjacent_list
        self.edge_list = edge_list
        self.q_classes = q_classes
        self.q_args = q_args
        self.max_agents = float(max_agents)

        self.sim_jobs = sim_jobs
        self.transition_proba = transition_proba

    def create_env(self):
        """
        Creates the queue network environment from the configured adjacency and edge lists.

        This method initializes the graph structure and the queue network with specified classes and arguments.
        """
        self.g = adjacency2graph(
            adjacency=self.adja_list, edge_type=self.edge_list, adjust=2
        )
        self.queueing_network = QueueNetwork(
            g=self.g,
            q_classes=self.q_classes,
            q_args=self.q_args,
            max_agents=self.max_agents,
        )
        self.queueing_network.set_transitions(self.transition_proba)

    def run_simulation(self, num_events=50, collect_data=True):
        """
        Runs the simulation of the queue network for a given number of events.

        Parameters:
        - num_events (int): The number of events to simulate.
        - collect_data (bool): Specifies whether to collect and store data during simulation.

        Returns:
        - dict: Collected data about agents if data collection is enabled; otherwise, None.
        """
        self.queueing_network.initial()
        if collect_data:
            self.queueing_network.start_collecting_data()
            self.queueing_network.simulate(n=num_events)
            self.agent_data = self.queueing_network.get_agent_data()  # check the output
