import networkx as nx
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict


class LouvainSampler(RegularSampler):
    def __init__(self, G: nx.Graph, resolution: float = 1):
        self.G = G
        self.resolution = resolution
        self.communities_number = None

    def sample_qubo_to_dict(self) -> dict:
        communities = nx.community.louvain_communities(
            self.G, resolution=self.resolution
        )
        self.communities_number = len(communities)
        result = communities_to_dict(communities)
        return result

    def sample_qubo_to_list(self) -> list:
        communities = nx.community.louvain_communities(
            self.G, resolution=self.resolution
        )
        self.communities_number = len(communities)
        communities = list(map(list, communities))
        return communities
