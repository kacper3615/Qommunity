import networkx as nx
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict


class LouvainSampler(RegularSampler):
    def __init__(self, G: nx.Graph, use_weights: bool = True, resolution: float = 1):
        self.G = G
        self.resolution = resolution
        self.communities_number = None
        self.weight = "weight" if use_weights else None

    def sample_qubo_to_dict(self) -> dict:
        communities = nx.community.louvain_communities(
            self.G, weight=self.weight, resolution=self.resolution
        )
        self.communities_number = len(communities)
        result = communities_to_dict(communities)
        return result

    def sample_qubo_to_list(self) -> list:
        communities = nx.community.louvain_communities(
            self.G, weight=self.weight, resolution=self.resolution
        )
        self.communities_number = len(communities)
        communities = list(map(list, communities))
        return communities
