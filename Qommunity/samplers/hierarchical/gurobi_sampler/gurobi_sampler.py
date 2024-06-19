from QHyper.solvers.classical.gurobi import Gurobi
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
from ..hierarchical_sampler import HierarchicalSampler
import networkx as nx


class GurobiSampler(HierarchicalSampler):
    def __init__(
        self,
        G: nx.Graph,
        resolution: float = 1,
        community: list = None,
        mip_gap: float | None = None,
        suppress_output: bool = True,
        threads: int = 1,
    ) -> None:
        if not community:
            community = [*range(G.number_of_nodes())]

        self.G = G
        self.resolution = resolution

        network = Network(G, resolution=resolution, community=community)
        problem = CommunityDetectionProblem(
            network, communities=2, one_hot_encoding=False
        )

        model_id = G.name + str(GurobiSampler.instance_counter)
        self.gurobi = Gurobi(
            problem=problem,
            model_name=model_id,
            mip_gap=mip_gap,
            suppress_output=suppress_output,
            threads=threads,
        )

    def sample_qubo_to_dict(self) -> dict:
        return self.gurobi.solve()

    def string_to_dict(s: str, prefix: str = "x") -> dict:
        result = {f"{prefix}{i}": int(s[i]) for i in range(len(s))}
        return result
