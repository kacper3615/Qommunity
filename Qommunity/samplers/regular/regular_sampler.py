import networkx as nx
from abc import ABC, abstractmethod


class RegularSampler(ABC):
    @abstractmethod
    def __init__(
        self, G: nx.Graph, time: float, community: list = None, resolution: float = 1
    ):
        self.G = G
        self.resolution = resolution
        self.time = time
        self.communities_number = None

    @abstractmethod
    def sample_qubo_to_dict(self) -> dict:
        pass

    @abstractmethod
    def sample_qubo_to_list(self) -> list:
        pass
