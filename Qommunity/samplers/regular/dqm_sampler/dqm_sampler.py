from QHyper.solvers.quantum_annealing.dqm import DQM
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..regular_sampler import RegularSampler
from ...utils import communities_to_list


class DQMSampler(RegularSampler):
    def __init__(
        self,
        G: nx.Graph,
        time: float,
        communities: int = 2,
        resolution: float = 1,
        community: list = None,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.time = time
        self.resolution = resolution
        self.communities_number = communities

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(network, communities=communities)
        self.dqm = DQM(problem=problem, time=time)

    def sample_qubo_to_dict(self) -> dict:
        sample = self.dqm.solve().first.sample
        return sample

    def sample_qubo_to_list(self) -> list:
        sample = self.dqm.solve().first.sample
        communities = communities_to_list(sample, self.communities_number)
        result = []
        for community in communities:
            result.append([int(x[1:]) for x in community])

        return result
