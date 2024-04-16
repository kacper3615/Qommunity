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
        print(f"\nresults: {results}\n")

        # We might need to permutate the communities order
        if results != ground_communities:
            results = get_communities_list_in_order(results, ground_communities)
            if results is None:
                print(f"\n\nsearcher: {searcher}, results: {results}\n\n")
            assert results is not None
        assert results == ground_communities


def _test_hierarchical_searcher(
    hierarchical_searcher: HierarchicalCommunitySearcher, ground_communities: list
):
    results = hierarchical_searcher.hierarchical_community_search()
    # We might need to permutate the results order
    if results != ground_communities:
        results = get_communities_list_in_order(results, ground_communities)
        assert results is not None
    assert results == ground_communities


def test_searchers_for_sample_graphs(sample_G0, sample_G1, sample_G2):
    G0, ground_communities_G0 = sample_G0
    G1, ground_communities_G1 = sample_G1
    G2, ground_communities_G2 = sample_G2
    # graphs = [G0, G1, G2]
    # grounds = [ground_communities_G0, ground_communities_G1, ground_communities_G2]
    # louvain_resolutions = [1, 1, 1]
    graphs = [G0, G1]
    grounds = [ground_communities_G0, ground_communities_G1]
    louvain_resolutions = [1, 1]

    for i, (G, ground_communities, resolution) in enumerate(zip(graphs, grounds, louvain_resolutions)):
        expected_communities_number = len(ground_communities)

        advantage = AdvantageSampler(G, 5)
        bayan = BayanSampler(G)
        leiden = LeidenSampler(G)
        # dqm = DQMSampler(G, 5, communities=expected_communities_number)
        louvain = LouvainSampler(G, resolution=resolution)

        # samplers = [louvain, dqm, bayan, leiden]
        samplers = [louvain, bayan, leiden]
        hierarchical_searcher = HierarchicalCommunitySearcher(advantage)
        regular_searchers = [CommunitySearcher(sampler) for sampler in samplers]

        # _test_regular_searchers(regular_searchers, ground_communities)
        # _test_hierarchical_searcher(hierarchical_searcher, ground_communities)
        for searcher in regular_searchers:
            results = searcher.community_search()
            print(f"\nresults: {results}, ground_communities: {ground_communities}\n")

            # We might need to permutate the communities order
            if results != ground_communities:
                results = get_communities_list_in_order(results, ground_communities)
                if results is None:
                    print(f"\n\ni: {i}, results: {results}\n\n")
                assert results is not None
            assert results == ground_communities

def communities_list_to_dict(communities: list) -> dict:
    communities_dict = {}
    nodes = []
    for i, comm in enumerate(communities):
        for node in comm:
            nodes.append(node)
            communities_dict[node] = i
    # Assert each node is in exactly one community
    # assert list(set(nodes)) == nodes
    print(f"list(set(nodes)): {list(set(nodes))}")
    print(f"nodes: {nodes}")
    return {
            str(k): communities_dict[k]
            for k in nodes
            # if str(k) in communities_dict
        }


def get_communities_list_in_order(
    communities_results: list, expected_communities_order: list
) -> list | None:
    in_order = None
    # Permutate community divisions
    permutations = [[*c] for c in itertools.permutations(communities_results)]
    for permutation in permutations:
        if permutation == expected_communities_order:
            in_order = permutation
    return in_order


if __name__ == "__main__":
    # test_sample_graph_G0()
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (5, 3), (5, 0)])
    c1 = [0, 1, 2]
    c2 = [3, 4, 5]
    c = [c1, c2]

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

    cc = [c1, c2, c3, c4]
    G_bis = nx.from_numpy_array(A_ij_bis)

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
    ccc = [c1, c2, c3, c4]
    G_bis_bis = nx.from_numpy_array(A_ij_bis_bis)

    communities_results = [[0, 1, 2], [3, 4, 5], [8, 6, 7], [9, 10, 11]]
    expected_communities_order = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    a = communities_list_to_dict(communities_results)
    b = communities_list_to_dict(expected_communities_order)

    print(a, "\n\n")
    print(b, "\n\n")

    print(a == b)

    # in_order = None
    # # Permutate community divisions
    # permutations = [[*c] for c in itertools.permutations(communities_results)]
    # for permutation in permutations:
    #     # print(permutation)
    #     if permutation == expected_communities_order:
    #         in_order = permutation
    # if in_order is not None:
    #     # return in_order
    #     pass

    # for permutation in permutations:
    #     for community in permutation:
    #         nodes_perm = [[*n] for n in itertools.permutations(community)]
    #         for perm in nodes_perm:
    #             print(perm)

    res = {}
    # for comm in communities_results:




        
    # print(f"\n\n\n {in_order}")




    # test_searchers_for_sample_graphs((G, c), (G_bis, cc), (G_bis_bis, ccc))
