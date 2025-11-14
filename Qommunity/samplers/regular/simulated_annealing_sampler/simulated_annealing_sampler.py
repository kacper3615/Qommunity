from typing import Any
import random

import networkx as nx
from simanneal import Annealer

from QHyper.problems.community_detection import Network, CommunityDetectionProblem

from ..regular_sampler import RegularSampler
from ...utils import communities_to_list


class CommunityDetectionAnnealer(Annealer):
    def __init__(
        self,
        num_nodes: int,
        num_communities: int,
        B: Any,
        initial_state: dict[int, int]
    ) -> None:
        super().__init__(initial_state)
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        self.B = B

    def move(self):
        node_id = random.randint(0, self.num_nodes - 1)
        new_community = random.randint(0, self.num_communities - 1)
        self.state[node_id] = new_community

    def energy(self):
        modularity = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.state[i] == self.state[j]:
                    modularity += self.B[i, j]
        return -modularity


class SimulatedAnnealingSampler(RegularSampler):
	def __init__(
		self,
		G: nx.Graph,
		communities: int,
		time: float = 50000,
		resolution: float = 1.0,
		community: list | None = None,
		Tmax: float = 25000.0,
		Tmin: float = 2.5,
		steps: int | None = None,
		updates: int = 100,
	) -> None:
		if community is None:
			community = [*range(G.number_of_nodes())]

		self.G = G
		self.time = time # it isn't necessary, can be any value
		self.resolution = resolution
		self.communities_number = communities
		self.Tmax = Tmax
		self.Tmin = Tmin
		self.updates = updates

		network = Network(
			G,
			resolution=resolution,
			community=community,
		)
		self.problem = CommunityDetectionProblem(
			network,
			communities=communities,
			one_hot_encoding=False,
		)

		self.steps = steps if steps is not None else int(max(time, 1))
		self.num_nodes = len(self.problem.community)
		self.num_communities = self.problem.cases
		self.B = self.problem.B

	def _generate_initial_state(self) -> dict[int, int]:
		return {i: random.randint(0, self.num_communities - 1) for i in range(self.num_nodes)}

	def _create_annealer(self, initial_state: dict[int, int]) -> Annealer:
		return CommunityDetectionAnnealer(
            self.num_nodes,
            self.num_communities,
            self.B,
            initial_state
        )

	def _solve(self) -> dict[int, int]:
		initial_state = self._generate_initial_state()
		annealer = self._create_annealer(initial_state)

		annealer.Tmax = self.Tmax
		annealer.Tmin = self.Tmin
		annealer.steps = self.steps
		annealer.updates = self.updates

		final_state, _ = annealer.anneal()
		return final_state

	def sample_qubo_to_dict(self) -> dict:
		final_state = self._solve()
		return {f"x{node_id}": community_id for node_id, community_id in final_state.items()}

	def sample_qubo_to_list(self) -> list:
		sample = self.sample_qubo_to_dict()
		communities = communities_to_list(sample, self.communities_number)
		return [[int(node[1:]) for node in community] for community in communities]
