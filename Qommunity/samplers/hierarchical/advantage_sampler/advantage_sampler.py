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
        use_clique_embedding: bool = False,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution
        self.version = version
        self.region = region
        self.num_reads = num_reads
        self.chain_strength = chain_strength
        self.use_clique_embedding = use_clique_embedding

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )
        self.advantage = Advantage(
            problem=problem,
            version=version,
            region=region,
            num_reads=num_reads,
            chain_strength=chain_strength,
            use_clique_embedding=use_clique_embedding,
        )

    def sample_qubo_to_dict(self) -> dict:
        sample = self.advantage.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("x")],
            key=lambda x: int(x[1:]),
        )
        community = sample.probabilities[variables][0]

        return dict(zip(variables, community))

    def update_community(self, community: list) -> None:
        self.__init__(
            self.G,
            self.resolution,
            community,
            self.version,
            self.region,
            self.num_reads,
            self.chain_strength,
            self.use_clique_embedding,
        )
