import networkx as nx
from samplers.sampler import Sampler


class LouvainSampler(Sampler):
    def __init__(self, G: nx.Graph, time: float, resolution: float = 0.5, community: list = None):
        self.G = G
        self.resolution = resolution
        self.community = community

    def sample_qubo_to_dict(self) -> dict: 
        communities = nx.community.louvain_communities(self.G, resolution=self.resolution)
        result = {}
        for i in range(len(communities)):
            for j in communities[i]:
                result[f"x{j}"] = i
        return result