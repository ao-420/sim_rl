import sys
from pathlib import Path

# Get the absolute path of the parent directory (i.e., the root of your project)
root_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to sys.path
sys.path.append(str(root_dir))

import pytest
from queue_env.queueing_network import *  # Adjust import based on your project structure
from queueing_tool.queues.queue_servers import *


@pytest.fixture
def queue_network():
    """
    Pytest fixture to create and return a new instance of Queue_network class.
    This setup is used for testing functions that require a Queue_network object.
    """
    return Queue_network()


def test_process_config(queue_network, tmp_path):
    """
    Test the `process_config` method of the `Queue_network` class to ensure it correctly
    parses and loads parameters from a YAML configuration file.

    Parameters:
    - queue_network: Fixture that provides an instance of the Queue_network class.
    - tmp_path: Pytest built-in fixture that provides a temporary directory unique to the test invocation.
    """
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "config_sample.yml"
    sample_config = """
    lambda_list: [0.1, 0.2]
    miu_list: [1.0, 2.0]
    active_cap: 5.0
    deactive_cap: 10.0
    adjacent_list: {1: [2, 3], 2: [3]}
    buffer_size: [10, 20, 30]
    """
    p.write_text(sample_config)

    (
        lambda_list,
        miu_list,
        active_cap,
        deactive_cap,
        adjacent_list,
        buffer_size_for_each_queue,
        transition_proba,
    ) = queue_network.process_config(str(p))

    assert lambda_list == [0.1, 0.2], "Incorrect lambda_list parsed from config"
    assert miu_list == [1.0, 2.0], "Incorrect miu_list parsed from config"
    assert active_cap == 5.0, "Incorrect active_cap parsed from config"
    assert deactive_cap == 10.0, "Incorrect deactive_cap parsed from config"
    assert adjacent_list == {
        1: [2, 3],
        2: [3],
    }, "Incorrect adjacent_list parsed from config"
    assert buffer_size_for_each_queue == [
        10,
        20,
        30,
    ], "Incorrect buffer_size parsed from config"


@pytest.fixture
def sample_network_input():
    """
    Pytest fixture to provide sample input parameters for testing the queue network initialization.
    This includes arrival rates, service rates, queue classes, and other parameters required for setup.

    Returns:
    A tuple containing all the necessary parameters to initialize and test the queue network.
    """
    arrival_rate = [0.1, 0.2]
    miu_list = [1.0, 2.0]
    q_classes = {1: QueueServer, 2: LossQueue}
    q_args = {
        1: {"service_f": lambda t: t + 1, "qbuffer": 10},
        2: {"service_f": lambda t: t + 2, "qbuffer": 20},
    }
    adjacent_list = {1: [2, 3], 2: [3]}
    edge_list = {1: {2: 1, 3: 2}, 2: {3: 3}}
    transition_proba = {1: {2: 0.5, 3: 0.5}, 2: {3: 1.0}}
    max_agents = float("inf")
    sim_jobs = 100

    return (
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba,
        max_agents,
        sim_jobs,
    )


def test_queue_network_initialization(sample_network_input):
    """
    Test the initialization of the `Queue_network` class to verify that all attributes
    are set correctly based on provided input parameters.

    Parameters:
    - sample_network_input: Fixture that provides sample input parameters for initializing the queue network.
    """
    network = Queue_network()

    # Unpack the sample input parameters
    (
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba,
        max_agents,
        sim_jobs,
    ) = sample_network_input

    # Process the input parameters to initialize the network
    network.process_input(
        arrival_rate,
        miu_list,
        q_classes,
        q_args,
        adjacent_list,
        edge_list,
        transition_proba,
        max_agents,
        sim_jobs,
    )

    # Now check if the network has been initialized as expected
    # This might involve checking various attributes of the network object
    assert network.lamda == arrival_rate, "Arrival rate not set correctly"
    assert network.miu == miu_list, "Service rates not set correctly"
    assert network.q_classes == q_classes, "Queue classes not set correctly"
    assert network.adja_list == adjacent_list, "adja_list not set correctly"
    assert network.edge_list == edge_list, "edge_list not set correctly"
    assert network.q_classes == q_classes, "q_classes not set correctly"
    assert network.q_args == q_args, "q_args not set correctly"


# Use this to run tests if you're executing the script directly
if __name__ == "__main__":
    pytest.main()
