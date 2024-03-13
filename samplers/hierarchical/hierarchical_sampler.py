import networkx as nx
from abc import ABC, abstractmethod

class HierarchicalSampler(ABC):
    @abstractmethod
    def __init__(self, G: nx.Graph, community: list = None):
        self.G = G

    @abstractmethod
    def sample_qubo_to_dict(self) -> dict:
        pass