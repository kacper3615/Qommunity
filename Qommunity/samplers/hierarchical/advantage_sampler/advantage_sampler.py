from QHyper.solvers.quantum_annealing.advantage import Advantage
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..hierarchical_sampler import HierarchicalSampler


class AdvantageSampler(HierarchicalSampler):
    def __init__(
        self, G: nx.Graph, resolution: float = 1, community: list = None, num_reads: int = 100
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )
        self.num_reads = num_reads # for test purposes, to delete later
        self.advantage = Advantage(problem=problem, num_reads=num_reads)

    def sample_qubo_to_dict(self) -> dict:
        return self.advantage.solve().first.sample

    def string_to_dict(s: str, prefix: str = "x") -> dict:
        result = {f"{prefix}{i}": int(s[i]) for i in range(len(s))}
        return result
