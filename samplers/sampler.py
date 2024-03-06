import networkx as nx
from abc import ABC, abstractmethod

class Sampler(ABC):
    @abstractmethod
    def __init__(self, G: nx.Graph, time: float, resolution: float = 1, community: list = None):
        self.G = G
        self.time = time
        self.resolution = resolution

    @abstractmethod
    def sample_qubo_to_dict(self) -> dict:
        pass