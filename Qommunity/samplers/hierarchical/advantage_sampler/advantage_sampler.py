from QHyper.solvers.quantum_annealing.dwave.advantage import Advantage
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..hierarchical_sampler import HierarchicalSampler
import numpy as np


class AdvantageSampler(HierarchicalSampler):
    def __init__(
        self,
        G: nx.Graph,
        resolution: float = 1,
        community: list | None = None,
        use_weights: bool = True,
        version: str = "Advantage_system5.4",
        region: str = "eu-central-1",
        num_reads: int = 100,
        chain_strength: float | None = None,
        use_clique_embedding: bool = False,
        elapse_times: bool = False,
        return_sampleset_metadata: bool = True,
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
        self._use_weights = use_weights
        self.elapse_times = elapse_times
        self.return_sampleset_metadata = return_sampleset_metadata

        weight = "weight" if use_weights else None
        network = Network(G, resolution=resolution, weight=weight, community=community)
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
            elapse_times=elapse_times
        )

    def sample_qubo_to_dict(self, return_sampleset_metadata: bool | None = None) -> dict:
        if return_sampleset_metadata:
            sample = self.advantage.solve(return_metadata=self.return_sampleset_metadata)
        else:
            sample = self.advantage.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("x")],
            key=lambda x: int(x[1:]),
        )
        community = sample.probabilities[variables][0]

        result = dict(zip(variables, community))

        if return_sampleset_metadata:
            return result, sample.sampleset_info
        return result

    def update_community(self, community: list) -> None:
        self.__init__(
            self.G,
            self.resolution,
            community,
            self._use_weights,
            self.version,
            self.region,
            self.num_reads,
            self.chain_strength,
            self.use_clique_embedding,
            self.elapse_times,
            self.return_sampleset_metadata
        )
