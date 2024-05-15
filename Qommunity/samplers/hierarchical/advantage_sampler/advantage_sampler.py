from QHyper.solvers.advantage import Advantage
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..hierarchical_sampler import HierarchicalSampler


class AdvantageSampler(HierarchicalSampler):
    def __init__(
        self, G: nx.Graph, resolution: float = 1, community: list = None
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )
        self.advantage = Advantage(problem=problem)

    def sample_qubo_to_dict(self) -> dict:
        return self.advantage.solve().first.sample
