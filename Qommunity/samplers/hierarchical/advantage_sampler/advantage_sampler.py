from QHyper.solvers.quantum_annealing.advantage import Advantage
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..hierarchical_sampler import HierarchicalSampler


class AdvantageSampler(HierarchicalSampler):
    def __init__(
        self,
        G: nx.Graph,
        resolution: float = 1,
        community: list = None,
        version: str = "Advantage_system5.4",
        region: str = "eu-central-1",
        num_reads: int = 1,
        chain_strength: float | None = None,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )
        self.advantage = Advantage(problem, version, region, num_reads, chain_strength)

    def sample_qubo_to_dict(self) -> dict:
        sample = self.advantage.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("x")],
            key=lambda x: int(x[1:]),
        )
        sample_communities = sample.probabilities[variables]

        best_modularity, best_community = 0, []

        for community in sample_communities:
            communities = [[], [], []]  # c0, c1, rest of the nodes
            community_dictonary = dict(zip(variables, community))
            indices = [int(var[1:]) for var in variables]

            for i in range(self.G.number_of_nodes()):
                if i in indices:
                    communities[community_dictonary[f"x{i}"]].append(i)
                else:
                    communities[2].append(i)

            modularity = nx.community.modularity(
                G=self.G,
                communities=communities,
                resolution=self.resolution,
            )

            if modularity > best_modularity:
                best_modularity, best_community = modularity, community

        return dict(zip(variables, best_community))
