import numpy as np
import networkx as nx
import matplotlib.pylab as plt

import pytest

from samplers.hierarchical.advantage_sampler import AdvantageSampler
from samplers.regular.dqm_sampler import DQMSampler
from samplers.regular.louvain_sampler import LouvainSampler
from searchers.community_searcher import CommunitySearcher
from samplers.regular.bayan_sampler import BayanSampler
from samplers.regular.leiden import LeidenSampler
from searchers.hierarchical_community_searcher.hierarchical_community_searcher import HierarchicalCommunitySearcher



@pytest.fixture()
def sample_G0():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 3), (5, 0)])
    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c = [c1, c2]

    return G, c


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


def test_sample_graph_G0(sample_G0):
    G, ground_communities = sample_G0

    expected_communities_number = len(ground_communities)

    advantage = AdvantageSampler(G, 5)
    bayan = BayanSampler(G)
    leiden = LeidenSampler(G)
    dqm = DQMSampler(G, 5, communities=expected_communities_number)
    louvain = LouvainSampler(G, 1)

    classical_samplers = [louvain, dqm, bayan, leiden]
    samplers = [advantage] + classical_samplers
    hierarchical_searcher = HierarchicalCommunitySearcher(advantage)
    classical_searchers = [CommunitySearcher(sampler) for sampler in classical_samplers]
    searchers = [hierarchical_searcher] + classical_searchers

    # Test classical searcher
    for searcher in classical_searchers:
        results = searcher.community_search()

        assert len(results) == expected_communities_number
        assert results == ground_communities

    # Test advantage

    
    

    # bayan_searcher = CommunitySearcher(bayan)
    # leiden_searcher = CommunitySearcher(leiden)


    # searcher = HierarchicalCommunitySearcher(advantage)
    # bayan_res = bayan_searcher.community_search()

def test_G():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 3), (5, 0)])
    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c = [c1, c2]
    print(c)


if __name__ == "__main__":
    test_G()
