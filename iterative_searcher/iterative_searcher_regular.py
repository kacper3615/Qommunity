import warnings
from Qommunity.searchers.community_searcher import CommunitySearcher
import networkx as nx
from time import time
from tqdm import tqdm
import numpy as np


class ResolutionValueWarning(Warning):
    message = (
        "The resolution passed to the objective function differs from"
        " the resolution used to calculate the modularity score."
    )

    def __init__(self):
        super().__init__(ResolutionValueWarning.message)


def warn(message, category, filename, lineno, file=None, line=None):
    tqdm.write(f"Warning: {str(message)}")


warnings.showwarning = warn

warnings.simplefilter("always", ResolutionValueWarning)


class IterativeSearcherRegular:
    def __init__(self, searcher: CommunitySearcher) -> None:
        self.searcher = searcher

    def run(
        self,
        num_runs: int,
        score_resolution: float = 1,
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

        if (
            hasattr(self.searcher.sampler, "resolution")
            and score_resolution != self.searcher.sampler.resolution
        ):
            warnings.warn(ResolutionValueWarning())

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.community_search(**kwargs)
            times[iter] = time() - elapsed

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G, result, resolution=score_resolution
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
