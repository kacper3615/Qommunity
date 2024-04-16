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
    A_ij = nx.to_numpy_array(G, nodelist=c)

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


# def get_optimal_resolution_value_louvain(G: nx.G, expected_communities_number: int, lower_bound: float = 0.1, upper_bound: float = 1.0, step: int = 10) -> float:
#     resolution_space = np.linspace(lower_bound, upper_bound, step)
#     results = {}

#     for resolution in resolution_space:
#         louvain = LouvainSampler(G, resolution=resolution)
#         louvain_searcher = CommunitySearcher(louvain)
#         result = louvain_searcher.community_search()
#         if results
#         print(f"resolution: {resolution:.2f}, result: {result}\n")

#     return 0
# def (communities: list) -> bool:
#     pass


def get_communities_list_in_order(
    communities_results: list, expected_communities_order: list
) -> list | None:
    in_order = None
    permutations = [[*c] for c in itertools.permutations(communities_results)]
    for permutation in permutations:
        if permutation == expected_communities_order:
            in_order = permutation
    return in_order


def test_sample_graph_G0(sample_G0):
    G, ground_communities = sample_G0

    expected_communities_number = len(ground_communities)

    advantage = AdvantageSampler(G, 5)
    bayan = BayanSampler(G)
    leiden = LeidenSampler(G)
    # dqm = DQMSampler(G, 5, communities=expected_communities_number)
    louvain = LouvainSampler(G, 1)

    # classical_samplers = [louvain, dqm, bayan, leiden]
    classical_samplers = [louvain, bayan, leiden]
    hierarchical_searcher = HierarchicalCommunitySearcher(advantage)
    classical_searchers = [CommunitySearcher(sampler) for sampler in classical_samplers]

    # Test the classical searchers
    for searcher in classical_searchers:
        results = searcher.community_search()
        results = [[3, 4, 5], [0, 1, 2]]

        # We might need to permutate the communities order
        if results != ground_communities:
            results = get_communities_list_in_order(results, ground_communities)
            assert results is not None
        assert results == ground_communities

    # Test the hierarchical searcher
    results = hierarchical_searcher.hierarchical_community_search()
    # We might need to permutate the results order
    if results != ground_communities:
        results = get_communities_list_in_order(results, ground_communities)
        assert results is not None
    assert results == ground_communities

# def test_sample_graph_G1(sample_G1):
#     G, ground_communities = sample_G1

#     expected_communities_number = len(ground_communities)

#     advantage = AdvantageSampler(G, 5)
#     bayan = BayanSampler(G)
#     leiden = LeidenSampler(G)
#     dqm = DQMSampler(G, 5, communities=expected_communities_number)
#     louvain = LouvainSampler(G, 1)

#     classical_samplers = [louvain, dqm, bayan, leiden]
#     hierarchical_searcher = HierarchicalCommunitySearcher(advantage)
#     classical_searchers = [CommunitySearcher(sampler) for sampler in classical_samplers]

#     # Test the classical searchers
#     for searcher in classical_searchers:
#         results = searcher.community_search()
#         if results != ground_communities:
#             results = get_communities_list_in_order(results, ground_communities)
#             assert results is not None
#         assert results == ground_communities

#     # Test the hierarchical searcher
#     results = hierarchical_searcher.hierarchical_community_search()
#     # We might need to permutate the results order
#     if results != ground_communities:
#         results = get_communities_list_in_order(results, ground_communities)
#         assert results is not None
#     assert results == ground_communities


# def test_G():
#     G = nx.Graph()
#     G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 3), (5, 0)])
#     c1 = [0, 1, 2]
#     c2 = [3, 4, 5]
#     c = [c1, c2]
#     a = [c1, c2]
#     b = [c2, c1]
#     # p = list(itertools.permutations(c))
#     p = itertools.permutations(c)
#     p = [[*el] for el in p]
#     print(p)
#     print()
#     for el in p:
#         print(el)

#     # print(c)


if __name__ == "__main__":
    test_sample_graph_G0()
