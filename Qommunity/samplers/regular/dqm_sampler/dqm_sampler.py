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
        cases: int,
        resolution: float = 1,
        use_weights: bool = True,
        community: list | None = None,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.time = time
        self.resolution = resolution
        self.communities_number = cases

        weights = "weight" if use_weights else None
        network = Network(G, resolution=resolution, weight=weights, community=community)
        problem = CommunityDetectionProblem(network, communities=cases)
        self.dqm = DQM(problem=problem, time=time, cases=cases)

    def sample_qubo_to_dict(self) -> dict:
        sample = self.dqm.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("s")],
            key=lambda s: int(s[1:]),
        )
        community = sample.probabilities[variables][0]

        return dict(zip(variables, community))

    def sample_qubo_to_list(self) -> list:
        sample = self.sample_qubo_to_dict()
        communities = communities_to_list(sample, self.communities_number)
        result = []
        for community in communities:
            result.append([int(x[1:]) for x in community])

        return result
