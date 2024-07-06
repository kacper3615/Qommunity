from QHyper.solvers.quantum_annealing.advantage import Advantage
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
import networkx as nx
from ..hierarchical_sampler import HierarchicalSampler

from QHyper.converter import Converter
from QHyper.solvers.quantum_annealing.advantage import convert_qubo_keys
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dwave.embedding.pegasus import find_clique_embedding
from time import time


class AdvantageSamplerCliqueEmbedding(HierarchicalSampler):
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
        self.version = version
        self.region = region
        self.num_reads = num_reads
        self.chain_strength = chain_strength

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )
        # self.advantage = Advantage(problem, version, region, num_reads, chain_strength)
        
        self.sampler = DWaveSampler(solver=self.version, region=self.region)
        qubo = Converter.create_qubo(problem, [])
        qubo_terms, offset = convert_qubo_keys(qubo)
        self.bqm = BinaryQuadraticModel.from_qubo(qubo_terms, offset=offset)

        # For test purposes measuring time
        start = time()
        self.embedding = find_clique_embedding(self.bqm.to_networkx_graph(), target_graph=self.sampler.to_networkx_graph())
        stop = time()
        elapsed = stop - start
        print(f"comm: {len(community)} elapsed: {elapsed}")
        
        self.embedding_composite = FixedEmbeddingComposite(self.sampler, embedding=self.embedding)


    def sample_qubo_to_dict(self) -> dict:
        sample = self.embedding_composite.sample(self.bqm, num_reads=self.num_reads)
        return sample.first.sample
    
    def update_community(self, community: list) -> None:
        self.__init__(
            self.G,
            self.resolution,
            community,
            self.version,
            self.region,
            self.num_reads,
            self.chain_strength,
        )
