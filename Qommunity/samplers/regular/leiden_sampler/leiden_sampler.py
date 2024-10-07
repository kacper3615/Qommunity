import networkx as nx
import igraph as ig
import leidenalg as la
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict


class LeidenSampler(RegularSampler):
    def __init__(self, G: nx.Graph, use_weights: bool = True, resolution: float = 1):
        self.G = G
        G_weights = list(nx.get_edge_attributes(G, "weight").values())
        self.weights = (
            list(nx.get_edge_attributes(G, "weight").values())
            if use_weights and G_weights
            else None
        )
        self.resolution = resolution

    def sample_qubo_to_dict(self) -> dict:
        communities = list(
            la.find_partition(
                ig.Graph.from_networkx(self.G),
                partition_type=la.RBConfigurationVertexPartition,
                weights=self.weights,
                resolution_parameter=self.resolution,
            )
        )
        self.communities_number = len(communities)
        sample = communities_to_dict(communities)
        return sample

    def sample_qubo_to_list(self) -> list:
        sample = list(
            la.find_partition(
                ig.Graph.from_networkx(self.G),
                partition_type=la.RBConfigurationVertexPartition,
                weights=self.weights,
                resolution_parameter=self.resolution,
            )
        )
        return sample
