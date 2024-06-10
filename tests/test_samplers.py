import pytest
import networkx as nx
import inspect
from Qommunity.samplers.regular.regular_sampler import RegularSampler
from Qommunity.samplers.regular.bayan_sampler import BayanSampler
from Qommunity.samplers.regular.dqm_sampler import DQMSampler
from Qommunity.samplers.regular.leiden_sampler import LeidenSampler
from Qommunity.samplers.regular.louvain_sampler import LouvainSampler
from Qommunity.samplers.hierarchical.hierarchical_sampler import HierarchicalSampler
from Qommunity.samplers.hierarchical.advantage_sampler import AdvantageSampler

def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
    return all_subclasses

@pytest.fixture(scope="session")
def regular_subclasses():
    return get_all_subclasses(RegularSampler)

@pytest.fixture(scope="session")
def hierarchical_subclasses():
    return get_all_subclasses(HierarchicalSampler)

@pytest.fixture()
def example_graph():
    return nx.karate_club_graph()

def test_regular_subclass_initialization(regular_subclasses, example_graph):
    for subclass in regular_subclasses:
        parameters = {'G': example_graph}
        subclass_parameters = inspect.signature(subclass.__init__).parameters
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            parameters['time'] = 5

        instance = subclass(**parameters)
        assert isinstance(instance, RegularSampler)
        assert instance.G == example_graph
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            assert instance.time == 5

def test_hierarchical_subclass_initialization(hierarchical_subclasses, example_graph):
    for subclass in hierarchical_subclasses:
        parameters = {'G': example_graph}
        subclass_parameters = inspect.signature(subclass.__init__).parameters
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            parameters['time'] = 5

        instance = subclass(**parameters)
        assert isinstance(instance, HierarchicalSampler)
        assert instance.G == example_graph
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            assert instance.time == 5

def test_regular_subclass_sample_qubo_to_dict(regular_subclasses, example_graph):
    for subclass in regular_subclasses:
        parameters = {'G': example_graph}
        subclass_parameters = inspect.signature(subclass.__init__).parameters
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            parameters['time'] = 5

        instance = subclass(**parameters)
        if hasattr(instance, 'sample_qubo_to_dict'):
            result = instance.sample_qubo_to_dict()
            assert isinstance(result, dict), "sample_qubo_to_dict should return a dict"

def test_regular_subclass_sample_qubo_to_list(regular_subclasses, example_graph):
    for subclass in regular_subclasses:
        parameters = {'G': example_graph}
        subclass_parameters = inspect.signature(subclass.__init__).parameters
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            parameters['time'] = 5

        instance = subclass(**parameters)
        if hasattr(instance, 'sample_qubo_to_list'):
            result = instance.sample_qubo_to_list()
            assert isinstance(result, list), "sample_qubo_to_list should return a list"
            assert all(isinstance(community, list) for community in result), "Each community in the result should be a list"

def test_hierarchical_subclass_sample_qubo_to_dict(hierarchical_subclasses, example_graph):
    for subclass in hierarchical_subclasses:
        parameters = {'G': example_graph}
        subclass_parameters = inspect.signature(subclass.__init__).parameters
        if 'time' in subclass_parameters and subclass_parameters['time'].default == inspect.Parameter.empty:
            parameters['time'] = 5

        instance = subclass(**parameters)
        if hasattr(instance, 'sample_qubo_to_dict'):
            result = instance.sample_qubo_to_dict()
            assert isinstance(result, dict), "sample_qubo_to_dict should return a dict"