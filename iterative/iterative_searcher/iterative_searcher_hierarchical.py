from Qommunity.searchers.hierarchical_community_searcher import (
    HierarchicalCommunitySearcher,
)
import networkx as nx
from time import time
from tqdm import tqdm
import numpy as np


class IterativeSearcherHierarchical:
    def __init__(self, searcher: HierarchicalCommunitySearcher) -> None:
        self.searcher = searcher

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
            saving_path = (
                f"{self.searcher.sampler.__class__.__name__}"
                + "-network_size_"
                + f"{self.searcher.sampler.G.number_of_nodes()}"
            )

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(**kwargs)
            times[iter] = time() - elapsed

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G, result
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}", modularities)
                np.save(f"{saving_path}_comms", communities)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        if elapse_times:
            return communities, modularities, times
        return communities, modularities

    def run_with_full_sampleset_info(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        iterative_verbosity: int = 0,
        **kwargs,
    ):

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = (
                f"{self.searcher.sampler.__class__.__name__}"
                + "-network_size_"
                + f"{self.searcher.sampler.G.number_of_nodes()}"
            )

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
                    self.searcher.sampler.G, communities_result
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = communities_result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}", modularities)
                np.save(f"{saving_path}_comms", communities)

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
