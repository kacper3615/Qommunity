from QHyper.solvers.classical.gurobi import Gurobi
from QHyper.problems.community_detection import Network, CommunityDetectionProblem
from ..hierarchical_sampler import HierarchicalSampler
import networkx as nx


class GurobiSampler(HierarchicalSampler):
    # instance_counter - only for tests purposes and to be deleted later
    instance_counter = 0
    def __init__(
        self, G: nx.Graph, resolution: float = 1, community: list = None, mip_gap: float | None = None, supress_output: bool = True
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
        
        # This is what I used in the notebook - commented here, because I updated my local version of QHyper
        # self.gurobi = Gurobi(problem=problem, model_name=model_id, mip_gap=mip_gap, suppress_output=supress_output)
        self.gurobi = Gurobi(problem=problem, model_name=model_id, mip_gap=mip_gap)
        GurobiSampler.instance_counter += 1


    def sample_qubo_to_dict(self) -> dict:
        return self.gurobi.solve()

    def string_to_dict(s: str, prefix: str = "x") -> dict:
        result = {f"{prefix}{i}": int(s[i]) for i in range(len(s))}
        return result