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
        cases: int = 2,
        resolution: float = 1,
        community: list = None,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.time = time
        self.resolution = resolution
        self.communities_number = cases

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(network, communities=cases)
        self.dqm = DQM(problem=problem, time=time, cases=cases)

    def sample_qubo_to_dict(self) -> dict:
        sample = self.dqm.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("s")],
            key=lambda s: int(s[1:]),
        )
        sample_communities = sample.probabilities[variables]

        best_modularity, best_community = 0, []

        for community in sample_communities:
            communities = []
            for i in range(self.communities_number):
                communities.append([])

            for i in range(self.G.number_of_nodes()):
                communities[community[i]].append(i)

            modularity = nx.community.modularity(
                G=self.G,
                communities=communities,
                resolution=self.resolution,
            )

            if modularity > best_modularity:
                best_modularity, best_community = modularity, community

        return dict(zip(variables, best_community))

    def sample_qubo_to_list(self) -> list:
        sample = self.sample_qubo_to_dict()
        communities = communities_to_list(sample, self.communities_number)
        result = []
        for community in communities:
            result.append([int(x[1:]) for x in community])

        return result
