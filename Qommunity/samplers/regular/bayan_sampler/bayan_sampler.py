import networkx as nx
import bayanpy
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict

class BayanSampler(RegularSampler):
    def __init__(self, G: nx.Graph, threshold: float = 0.001, time_allowed: int = 60, 
                 resolution: float = 1, community: list = None) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution
        self.threshold = threshold
        self.time_allowed = time_allowed

    def sample_qubo_to_dict(self) -> dict:
        _, _, communities, _, _ = bayanpy.bayan(self.G, self.threshold, self.time_allowed, self.resolution)
        self.communities_number = len(communities)
        sample = communities_to_dict(communities)
        return sample
    
    def sample_qubo_to_list(self) -> list:
        _, _, sample, _, _ = bayanpy.bayan(self.G, self.threshold, self.time_allowed, self.resolution)
        return sample
