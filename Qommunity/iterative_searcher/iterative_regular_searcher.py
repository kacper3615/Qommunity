from Qommunity.samplers.regular.regular_sampler import RegularSampler
from Qommunity.searchers.regular_searcher import RegularSearcher
import networkx as nx
from time import time
from tqdm import tqdm
import numpy as np


class IterativeRegularSearcher:
    def __init__(self, sampler: RegularSampler) -> None:
        self.sampler = sampler
        self.searcher = RegularSearcher(self.sampler)

    def _default_saving_path(self) -> str:
        return (
            f"{self.sampler.__class__.__name__}"
            + "-network_size_"
            + f"{self.sampler.G.number_of_nodes()}"
        )

    def run(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        elapse_times: bool = True,
        iterative_verbosity: int = 0,
        **kwargs,
    ):

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()
        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.community_search(**kwargs)
            times[iter] = time() - elapsed

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G,
                    result,
                    resolution=self.sampler.resolution,
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}_modularities", modularities)
                np.save(f"{saving_path}_communities", communities)
                if elapse_times:
                    np.save(f"{saving_path}_times", times)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        if elapse_times:
            return communities, modularities, times
        return communities, modularities
