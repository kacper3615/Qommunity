import numpy as np
import networkx as nx

import pytest
import itertools

from samplers.hierarchical.advantage_sampler import AdvantageSampler
from samplers.hierarchical.hierarchical_sampler import HierarchicalSampler
from samplers.regular.dqm_sampler import DQMSampler
from samplers.regular.louvain_sampler import LouvainSampler
from samplers.regular.regular_sampler import RegularSampler
from searchers.community_searcher import CommunitySearcher
from searchers.hierarchical_community_searcher.hierarchical_community_searcher import (
    HierarchicalCommunitySearcher,
)


@pytest.fixture()
def sample_G0():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 3), (5, 0)])
    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c = [c1, c2]

    return G, c


@pytest.fixture()
def sample_G1(sample_G0):
    G, c = sample_G0
    A_ij = nx.to_numpy_array(G, nodelist=list(itertools.chain(*c)))

    A_ij_bis = np.zeros((12, 12))
    A_ij_bis[:6, :6] = A_ij
    A_ij_bis[6:12, 6:12] = A_ij
    A_ij_bis[6, 5] = 1
    A_ij_bis[0, -1] = 1

    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c3 = [6, 7, 8]
    c4 = [9, 10, 11]

    c = [c1, c2, c3, c4]
    G_bis = nx.from_numpy_array(A_ij_bis)

    return G_bis, c


@pytest.fixture()
def sample_G2(sample_G0):
    G, c = sample_G0
    A_ij = nx.to_numpy_array(G, nodelist=list(itertools.chain(*c)))

    A_ij_bis_bis = np.zeros((12, 12))
    A_ij_bis_bis[:6, :6] = A_ij
    A_ij_bis_bis[6:12, 6:12] = A_ij
    A_ij_bis_bis[6, 5] = 1
    A_ij_bis_bis[5, 6] = 1
    A_ij_bis_bis[0, -1] = 1
    A_ij_bis_bis[-1, 0] = 1

    # Extra edge
    A_ij_bis_bis[0, 3] = 1
    A_ij_bis_bis[3, 0] = 1

    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c3 = [6, 7, 8]
    c4 = [9, 10, 11]
    c = [c1, c2, c3, c4]
    G_bis_bis = nx.from_numpy_array(A_ij_bis_bis)

    return G_bis_bis, c


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
    return all_subclasses


@pytest.fixture(scope="session")
def regular_samplers_subclasses():
    return get_all_subclasses(RegularSampler)


@pytest.fixture(scope="session")
def hierarchical_samplers_subclasses():
    return get_all_subclasses(HierarchicalSampler)


def test_hierarchical_searcher_subclass_for_sample_graphs(
    sample_G0, sample_G1, sample_G2, hierarchical_samplers_subclasses
):
    G0, ground_communities_G0 = sample_G0
    G1, ground_communities_G1 = sample_G1
    G2, ground_communities_G2 = sample_G2
    graphs = [G0, G1, G2]
    grounds = [ground_communities_G0, ground_communities_G1, ground_communities_G2]

    for G, ground_communities in zip(graphs, grounds):
        hierarchical_samplers = []
        for subclass in hierarchical_samplers_subclasses:
            if subclass is AdvantageSampler:
                hierarchical_samplers.append(subclass(G, time=5))
            else:
                hierarchical_samplers.append(subclass(G))

        hierarchical_searchers = [
            HierarchicalCommunitySearcher(sampler) for sampler in hierarchical_samplers
        ]
        for searcher in hierarchical_searchers:
            results = searcher.hierarchical_community_search()
            # Detected communities may appear in random order
            # so let's sort them to compare them
            results_sorted = sorted([sorted(community) for community in results])
            ground_communities_sorted = sorted(
                [sorted(community) for community in ground_communities]
            )
            assert results_sorted == ground_communities_sorted


def test_regular_searcher_subclass_for_sample_graphs(
    sample_G0, sample_G1, sample_G2, regular_samplers_subclasses
):
    G0, ground_communities_G0 = sample_G0
    G1, ground_communities_G1 = sample_G1
    G2, ground_communities_G2 = sample_G2
    graphs = [G0, G1, G2]
    grounds = [ground_communities_G0, ground_communities_G1, ground_communities_G2]
    louvain_resolutions = [1, 1, 1]

    for G, ground_communities, resolution in zip(graphs, grounds, louvain_resolutions):
        samplers = []
        for subclass in regular_samplers_subclasses:
            if subclass is DQMSampler:
                samplers.append(subclass(G, time=5, communities=len(ground_communities)))
                pass
            elif subclass is LouvainSampler:
                samplers.append(subclass(G, resolution=resolution))
            else:
                samplers.append(subclass(G))

        regular_searchers = [CommunitySearcher(sampler) for sampler in samplers]
        for searcher in regular_searchers:
            results = searcher.community_search()
            # Detected communities may appear in random order
            # so let's sort them to compare them
            results_sorted = sorted([sorted(community) for community in results])
            ground_communities_sorted = sorted(
                [sorted(community) for community in ground_communities]
            )
            assert results_sorted == ground_communities_sorted
