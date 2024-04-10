import pytest
import networkx as nx
from Qommunity.searchers.community_searcher import CommunitySearcher
from Qommunity.samplers.regular.louvain_sampler import LouvainSampler

@pytest.fixture
def regular_sampler():
    G = nx.karate_club_graph()
    sampler = LouvainSampler(G)
    return sampler

# Test inicjalizacji
def test_community_searcher_initialization(regular_sampler):
    community_searcher = CommunitySearcher(regular_sampler)
    assert community_searcher.sampler is regular_sampler

# Test metody community_search z verbosity=0 i return_list=True
def test_community_search_verbosity_0_return_list(regular_sampler, capsys):
    community_searcher = CommunitySearcher(regular_sampler)
    result = community_searcher.community_search(verbosity=0, return_list=True)
    captured = capsys.readouterr()
    assert result is not None
    assert captured.out == ""

# Test metody community_search z verbosity=2 i return_list=False
def test_community_search_verbosity_2_return_list_false(regular_sampler, capsys):
    community_searcher = CommunitySearcher(regular_sampler)
    community_searcher.community_search(verbosity=2, return_list=False)
    captured = capsys.readouterr()
    # Sprawdź, czy odpowiednie komunikaty zostały wydrukowane
    assert "Starting community detection" in captured.out
    assert "Stopping community detection" in captured.out

# Test metody _communities_to_list
def test_communities_to_list(regular_sampler):
    community_searcher = CommunitySearcher(regular_sampler)
    # Przygotuj mocka wyniku
    sample_result = {'a': 0,
                     'b': 0,
                     'c': 0,
                     'd': 1}
    regular_sampler.communities_number = 2
    result = community_searcher._communities_to_list(sample_result)
    # Sprawdź, czy wynik jest poprawny
    assert isinstance(result, list)
    assert len(result) == 2
    assert result == [['a', 'b', 'c'], ['d']]