import networkx as nx
import igraph as ig
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict


class InfomapSampler(RegularSampler):
    def __init__(self, G: nx.Graph, use_weights: bool = True, trials: int = 10):
        self.G = G
        G_weights = list(nx.get_edge_attributes(G, "weight").values())
        self.weights = (
            list(nx.get_edge_attributes(G, "weight").values())
            if use_weights and G_weights
            else None
        )
        self.trials = trials
        self.resolution = 1 # For compatibility 

    def sample_qubo_to_dict(self) -> dict:
        communities = list(
            ig.Graph.community_infomap(
                ig.Graph.from_networkx(self.G), 
                edge_weights=self.weights,
                trials=self.trials
            )
        )
        self.communities_number = len(communities)
        sample = communities_to_dict(communities)
        return sample

    def sample_qubo_to_list(self) -> list:
        sample = list(
            ig.Graph.community_infomap(
                ig.Graph.from_networkx(self.G), 
                edge_weights=self.weights,
                trials=self.trials
            )
        )
        return sample
