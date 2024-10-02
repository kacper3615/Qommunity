import networkx as nx
from abc import ABC, abstractmethod


class HierarchicalSampler(ABC):
    @abstractmethod
    def __init__(self, G: nx.Graph, resolution: float = 1, community: list = None):
        self.G = G
        self.resolution = resolution

    @abstractmethod
    def sample_qubo_to_dict(self) -> dict:
        pass

    @abstractmethod
    def update_community(self, community: list) -> None:
        pass
