import numpy as np
import networkx as nx
import matplotlib.pylab as plt

import pytest
import itertools

from samplers.hierarchical.advantage_sampler import AdvantageSampler
from samplers.regular.dqm_sampler import DQMSampler
from samplers.regular.louvain_sampler import LouvainSampler
from searchers.community_searcher import CommunitySearcher
from samplers.regular.bayan_sampler import BayanSampler
from samplers.regular.leiden import LeidenSampler
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


def _test_regular_searchers(regular_searchers: list, ground_communities: list):
    for searcher in regular_searchers:
        results = searcher.community_search()
        results_sorted = sorted([sorted(community) for community in results])
        ground_communities_sorted = sorted(
            [sorted(community) for community in ground_communities]
        )
        assert results_sorted == ground_communities_sorted


def _test_hierarchical_searcher(
    hierarchical_searcher: HierarchicalCommunitySearcher, ground_communities: list
):
    results = hierarchical_searcher.hierarchical_community_search()
    results_sorted = sorted([sorted(community) for community in results])
    ground_communities_sorted = sorted(
        [sorted(community) for community in ground_communities]
    )
    assert results_sorted == ground_communities_sorted


def test_searchers_for_sample_graphs(sample_G0, sample_G1, sample_G2):
    G0, ground_communities_G0 = sample_G0
    G1, ground_communities_G1 = sample_G1
    G2, ground_communities_G2 = sample_G2
    graphs = [G0, G1, G2]
    grounds = [ground_communities_G0, ground_communities_G1, ground_communities_G2]
    louvain_resolutions = [1, 1, 1]

    for G, ground_communities, resolution in zip(graphs, grounds, louvain_resolutions):
        advantage = AdvantageSampler(G, 5)
        bayan = BayanSampler(G)
        leiden = LeidenSampler(G)
        dqm = DQMSampler(G, 5, communities=len(ground_communities))
        louvain = LouvainSampler(G, resolution=resolution)

        samplers = [louvain, dqm, bayan, leiden]
        hierarchical_searcher = HierarchicalCommunitySearcher(advantage)
        regular_searchers = [CommunitySearcher(sampler) for sampler in samplers]

        _test_regular_searchers(regular_searchers, ground_communities)
        _test_hierarchical_searcher(hierarchical_searcher, ground_communities)
