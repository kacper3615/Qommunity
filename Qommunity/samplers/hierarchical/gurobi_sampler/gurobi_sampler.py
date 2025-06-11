from QHyper.solvers.classical.gurobi import Gurobi
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
from ..hierarchical_sampler import HierarchicalSampler
import networkx as nx


class GurobiSampler(HierarchicalSampler):
    def __init__(
        self,
        G: nx.Graph,
        resolution: float = 1,
        community: list | None = None,
        use_weights: bool = True,
        mip_gap: float | None = None,
        suppress_output: bool = True,
        threads: int = 0,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution
        self.mip_gap = mip_gap
        self.suppress_output = suppress_output
        self.threads = threads
        self._use_weights = use_weights

        weight = "weight" if use_weights else None
        network = Network(G, resolution=resolution, weight=weight, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )

        model_id = G.name
        self.gurobi = Gurobi(
            problem=problem,
            model_name=model_id,
            mip_gap=mip_gap,
            suppress_output=suppress_output,
            threads=threads,
        )

    def sample_qubo_to_dict(self) -> dict:
        sample = self.gurobi.solve()

        variables = sorted(
            [col for col in sample.probabilities.dtype.names if col.startswith("x")],
            key=lambda x: int(x[1:]),
        )
        community = sample.probabilities[variables][0]

        return dict(zip(variables, community))

    def string_to_dict(s: str, prefix: str = "x") -> dict:
        result = {f"{prefix}{i}": int(s[i]) for i in range(len(s))}
        return result

    def update_community(self, community: list) -> None:
        self.__init__(
            self.G,
            self.resolution,
            community,
            self._use_weights,
            self.mip_gap,
            self.suppress_output,
            self.threads,
        )
