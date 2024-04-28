import torch
import numpy as np
import pandas as pd
import queueing_tool as qt
import numpy as np
import os

from agents.ddpg_agent import DDPGAgent
from queue_env.queueing_network import *
from foundations.core_plotting import *
from queueing_tool.queues.queue_servers import *


def get_num_connections(adjacent_list):
    """
    Calculates the total number of connections and identifies exit nodes within the adjacency list of a network.

    Parameters:
    - adjacent_list (dict): A dictionary where keys are start nodes and values are lists of end nodes they connect to.

    Returns:
    - tuple: A tuple containing the total number of connections (int) and a list of exit nodes ([]).
    """
    exit_nodes = []
    for start_node in adjacent_list.keys():
        end_node_list = adjacent_list[start_node]
        for end_node in end_node_list:
            if end_node not in list(adjacent_list.keys()):
                if end_node not in exit_nodes:
                    exit_nodes.append(end_node)

    return exit_nodes


def load_config(env_param_filepath):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_filepath (str): The file path to the configuration YAML file.

    Returns:
    - dict: A dictionary containing the configuration parameters.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one directory to the MScDataSparqProject directory
    project_dir = os.path.dirname(script_dir)

    # Build the path to the configuration file
    abs_file_path = os.path.join(project_dir, env_param_filepath)

    with open(abs_file_path, "r") as env_param_file:
        config_params = yaml.load(env_param_file, Loader=yaml.FullLoader)

    # Convert lists to tuples
    config_params["entry_nodes"] = [
        tuple(node) for node in config_params["entry_nodes"]
    ]

    return config_params


def make_edge_list(adjacent_list, exit_nodes):
    """
    Creates an edge list with types for each connection based on the adjacency list and identified exit nodes.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.
    - exit_nodes (list): A list of nodes identified as exit points in the network.

    Returns:
    - dict: A dictionary representing the edge list, where keys are start nodes, and values are dictionaries of end nodes with their edge types.
    """
    edge_list = {}
    edge_type = 1
    for start_node in adjacent_list.keys():
        end_node_list = adjacent_list[start_node]

        connection_dict = {}
        for end_node in end_node_list:
            if end_node not in exit_nodes:
                connection_dict[end_node] = edge_type
                edge_type += 1
            else:
                connection_dict[end_node] = 0

        edge_list[start_node] = connection_dict

    return edge_list


def make_unique_edge_type(adjacent_list, edge_list):
    """
    Assigns a unique edge type to connections between nodes based on the adjacency and edge lists.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.
    - edge_list (dict): A dictionary representing the network's edge list, indicating connections between nodes.

    Returns:
    - dict: A dictionary where keys are node identifiers, and values are lists of unique edge types for edges ending at that node.
    """
    connection_info = get_connection_info(adjacent_list)
    edge_type_info = {}
    for end_node in connection_info.keys():
        start_node_list = connection_info[end_node]
        edge_type_list = []
        for start_node in start_node_list:
            edge_type = edge_list[start_node][end_node]
            edge_type_list.append(edge_type)
        edge_type_info[end_node] = edge_type_list

    return edge_type_info  # keys are node_id, values are the edge_types


def get_connection_info(adjacent_list):
    """
    Generates a dictionary mapping each node to a list of nodes that connect to it.

    Parameters:
    - adjacent_list (dict): A dictionary representing the network's adjacency list.

    Returns:
    - dict: A dictionary where keys are end nodes, and values are lists of start nodes that connect to these end nodes.
    """
    connection_info = {}
    for start_node in adjacent_list.keys():
        for end_node in adjacent_list[start_node]:
            connect_start_node_list = connection_info.setdefault(end_node, [])
            connect_start_node_list.append(start_node)
            connection_info[end_node] = connect_start_node_list

    return connection_info


def create_params(config_file):
    """
    Generate parameters for the queueing environment based on a configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Multiple return values including lists and dictionaries essential for creating the queueing environment.
    """

    config_params = load_config(config_file)

    miu_dict = config_params["miu_list"]
    adjacent_list = config_params["adjacent_list"]
    max_agents = config_params["max_agents"]
    sim_jobs = config_params["sim_jobs"]
    entry_nodes = config_params["entry_nodes"]
    exit_nodes = get_num_connections(adjacent_list)
    edge_list = make_edge_list(adjacent_list, exit_nodes)

    q_classes = create_q_classes(edge_list)

    edge_type_info = make_unique_edge_type(adjacent_list, edge_list)

    buffer_size_for_each_queue = config_params["buffer_size_for_each_queue"]
    q_args = create_q_args(
        edge_type_info,
        config_params,
        miu_dict,
        buffer_size_for_each_queue,
        exit_nodes,
        edge_list,
        q_classes,
    )

    arrival_rate = config_params["arrival_rate"]

    transition_proba_all = config_params["transition_proba_all"]

    return (
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


def get_entry_nodes(config_file):
    config_params = load_config(config_file)
    entry_nodes = [tuple(entry_node) for entry_node in config_params["entry_nodes"]]
    return entry_nodes


def create_q_classes(edge_list):
    """
    Creates a dictionary mapping queue identifiers to their corresponding queue class.

    Parameters:
    - num_queues (int): The number of queues to create classes for, excluding the null queue.

    Returns:
    - dict: A dictionary where keys are queue identifiers (starting from 1) and values are queue class types.
    """
    q_classes = {}
    for start_node in edge_list.keys():
        end_nodes_dict = edge_list[start_node]

        for end_node in end_nodes_dict.keys():
            edge_index = end_nodes_dict[end_node]
            if end_node in edge_list.keys():
                q_classes[edge_index] = LossQueue
            else:
                q_classes[edge_index] = NullQueue
    return q_classes


def get_node_tuple_from_edgetype(edge_list):
    """
    Creates a dictionary mapping each edge type to a list of tuples, where each tuple represents an edge
    from a source node to an end node.

    Parameters:
    - edge_list (dict): A dictionary where each key is a source node and its value is another dictionary.
      The nested dictionary's key is the end node and its value is the edge type.

    Returns:
    - dict: A dictionary where keys are edge types and values are lists of tuples (source_node, end_node)
      representing the edges of that type.
    """
    node_tuple_dict = {}

    for source_node, endnode_type_dict in edge_list.items():
        for end_node, edge_type in endnode_type_dict.items():
            if edge_type in node_tuple_dict:
                node_tuple_dict[edge_type].append((source_node, end_node))
            else:
                node_tuple_dict[edge_type] = [(source_node, end_node)]
    return node_tuple_dict


def get_node_id(edge_type, edge_type_info):
    """
    Finds and returns the node ID associated with a specific edge type.

    Parameters:
    - edge_type (any): The type of the edge for which the node ID is to be found.
    - edge_type_info (dict): A dictionary where each key is a node ID and its value is a list of edge types
      associated with that node.

    Returns:
    - any: The node ID for the specified edge type if found; otherwise, None if no node has the specified edge type.
    """
    for node in edge_type_info.keys():
        if edge_type in edge_type_info[node]:
            return node


def create_q_args(
    edge_type_info,
    config_params,
    miu_dict,
    buffer_size_for_each_queue,
    exit_nodes,
    edge_list,
    q_classes,
):
    """
    Constructs arguments for queue initialization based on the network configuration.

    Parameters:
    - edge_type_info (dict): Information about edge types for each node.
    - config_params (dict): Configuration parameters including service rates and buffer sizes.
    - miu_dict (dict): A dictionary mapping nodes to their service rates.
    - buffer_size_for_each_queue (dict): A dictionary mapping queue identifiers to their buffer sizes.
    - exit_nodes (list): A list of nodes identified as exit points in the network.

    Returns:
    - dict: A dictionary of queue arguments where keys are queue identifiers, and values are dictionaries of arguments needed for initializing each queue.
    """
    q_args = {}
    edge_type_lists = []
    for key in edge_type_info.keys():
        if key not in exit_nodes:
            values = edge_type_info[key]
            edge_type_lists += values

    node_tuple_by_edgetype = get_node_tuple_from_edgetype(edge_list)
    entry_node_encountered = 0
    env_entry_nodes = [tuple(item) for item in config_params["entry_nodes"]]

    for edge_type in edge_type_lists:
        queue_type = q_classes[edge_type]
        node_id = get_node_id(edge_type, edge_type_info)
        service_rate = miu_dict[node_id]

        if queue_type == LossQueue:
            if node_tuple_by_edgetype[edge_type][0] in config_params["entry_nodes"]:
                max_arrival_rate = config_params["arrival_rate"][entry_node_encountered]
                rate = (
                    lambda t: 0.1 * (max_arrival_rate)
                    + (1 - 0.1) * (max_arrival_rate) * np.sin(np.pi * t / 2) ** 2
                )
                q_args[edge_type] = {
                    "arrival_f": lambda t, rate=rate: poisson_random_measure(
                        t, rate, max_arrival_rate
                    ),
                    "service_f": lambda t, en=node_id: t + np.exp(miu_dict[en]),
                    "qbuffer": buffer_size_for_each_queue[edge_type],
                    "service_rate": service_rate,
                    "active_cap": float("inf"),
                    "active_status": True,
                }
                entry_node_encountered += 1
            else:
                q_args[edge_type] = {
                    "service_f": lambda t, en=node_id: t + np.exp(miu_dict[en]),
                    "qbuffer": buffer_size_for_each_queue[edge_type],
                    "service_rate": service_rate,
                    "active_cap": float("inf"),
                    "active_status": False,
                }

    return q_args


def create_queueing_env(config_file):
    """
    Create and configure a queueing environment based on a given configuration file.

    Parameters:
    - config_file (str): The file path to the environment configuration file.

    Returns:
    - Queue_network: An instance of the queueing environment.
    """
    (
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba_all,
        max_agents,
        sim_time,
    ) = create_params(config_file)

    q_net = Queue_network()
    q_net.process_input(
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba_all,
        max_agents,
        sim_time,
    )
    q_net.create_env()
    return q_net
