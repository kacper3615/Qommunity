from Qommunity.samplers.hierarchical.hierarchical_sampler import HierarchicalSampler
from Qommunity.searchers.hierarchical_community_searcher import (
    HierarchicalCommunitySearcher,
)
import networkx as nx
from time import time
from tqdm import tqdm
import numpy as np
import warnings


class MethodArgsWarning(Warning):
    def __init__(self, msg):
        super().__init__(msg)


# Warning format compatible with tqdm
def warn(message, category, filename, lineno, file=None, line=None):
    tqdm.write("Warning: {str(message)}")


warnings.showwarning = warn
warnings.simplefilter("always", MethodArgsWarning)


class IterativeSearcherHierarchical:
    def __init__(self, sampler: HierarchicalSampler) -> None:
        self.sampler = sampler
        self.searcher = HierarchicalCommunitySearcher(self.sampler)

    def default_saving_path(self) -> str:
        return (
                f"{self.sampler.__class__.__name__}"
                + "-network_size_"
                + f"{self.sampler.G.number_of_nodes()}"
            )

    def verify_kwargs(self, kwargs) -> dict:
        kwargs_unhandled = ["division_tree", "return_modularities"]
        kwargs_warning = []
        for kwarg in kwargs_unhandled:
            if kwarg in kwargs:
                kwargs.pop(kwarg, None)
                kwargs_warning.append(kwarg)
        if kwargs_warning:
            msg = ", ".join(kwargs_warning)
            warnings.warn(f"in order to get {msg} run " + " IterativeSearcher.run_with_sampleset_info()")

        return kwargs

    def run(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        elapse_times: bool = True,
        iterative_verbosity: int = 0,
        **kwargs,
    ):
        kwargs = self.verify_kwargs(kwargs)

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self.default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(**kwargs)
            times[iter] = time() - elapsed

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G, result, resolution=self.sampler.resolution
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}", modularities)
                np.save(f"{saving_path}_comms", communities)
                if elapse_times:
                    np.save(f"{saving_path}_times", times)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        if elapse_times:
            return communities, modularities, times
        return communities, modularities

    def run_with_sampleset_info(
        self,
        num_runs: int,
        score_resolution: float = 1,
        save_results: bool = True,
        saving_path: str | None = None,
        iterative_verbosity: int = 0,
        **kwargs,
    ):

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self.default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))
        division_modularities = np.empty((num_runs), dtype=object)
        division_trees = np.empty((num_runs), dtype=object)

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            (
                communities_result,
                div_tree,
                div_modularities,
            ) = self.searcher.hierarchical_community_search(
                return_modularities=True,
                division_tree=True,
                **kwargs,
            )
            times[iter] = time() - elapsed
            division_trees[iter] = div_tree
            division_modularities[iter] = div_modularities

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G,
                    communities_result,
                    resolution=score_resolution,
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = communities_result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}", modularities)
                np.save(f"{saving_path}_comms", communities)
                np.save(f"{saving_path}_times", times)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        dtypes = [
            ("communities", object),
            ("modularity", np.float_),
            ("time", np.float_),
            ("division_tree", object),
            ("division_modularities", object),
        ]
        sampleset = np.rec.fromarrays(
            [communities, modularities, times, division_trees, division_modularities],
            dtype=dtypes,
        )

        return sampleset
