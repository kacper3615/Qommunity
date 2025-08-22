from Qommunity.samplers.hierarchical.hierarchical_sampler import (
    HierarchicalSampler,
)
from Qommunity.searchers.hierarchical_searcher import (
    HierarchicalSearcher,
)
import networkx as nx
from time import time
from tqdm import tqdm
import numpy as np
import warnings
import pickle

from Qommunity.samplers.hierarchical.advantage_sampler import AdvantageSampler

METADATA_KEYARG = "return_metadata"


class MethodArgsWarning(Warning):
    def __init__(self, msg):
        super().__init__(msg)


# Warning format compatible with tqdm
def warn(message, category, filename, lineno, file=None, line=None):
    tqdm.write(f"Warning: {str(message)}")


warnings.showwarning = warn
warnings.simplefilter("always", MethodArgsWarning)


class IterativeHierarchicalSearcher:
    def __init__(self, sampler: HierarchicalSampler) -> None:
        self.sampler = sampler
        self.searcher = HierarchicalSearcher(self.sampler)

    def _default_saving_path(self) -> str:
        return (
            f"{self.sampler.__class__.__name__}"
            + "-network_size_"
            + f"{self.sampler.G.number_of_nodes()}"
        )

    def _verify_kwargs(self, kwargs) -> dict:
        kwargs_unhandled = ["division_tree", "return_modularities"]
        kwargs_warning = []
        for kwarg in kwargs_unhandled:
            if kwarg in kwargs:
                kwargs.pop(kwarg, None)
                kwargs_warning.append(kwarg)
        if kwargs_warning:
            msg = ", ".join(kwargs_warning)
            warnings.warn(
                f"in order to get {msg} run "
                + " IterativeSearcher.run_with_sampleset_info()"
            )

        return kwargs

    def _check_sampler_and_it_searcher_metadata_flags_compatibility(
        self, return_metadata_flag: bool
    ) -> bool:
        return self.sampler.return_metadata and return_metadata_flag

    def run(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        elapse_times: bool = True,
        iterative_verbosity: int = 0,
        return_metadata: bool = False,
        **kwargs,
    ):
        kwargs = self._verify_kwargs(kwargs)

        if return_metadata and not self.sampler.return_metadata:
            raise MethodArgsWarning(
                f"Set Advantage sampler's {METADATA_KEYARG} flag to True before running."
                + f" HierarchicalIterativeSearcher with {METADATA_KEYARG}."
            )

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))

        # List instead of samplesets_data = np.empty((num_runs), dtype=object)
        # To prevent jupyter notebook kernel crashes
        # as handling big objects is not efficient with numpy dtype=object arrs
        samplesets_data = []

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(**kwargs)
            times[iter] = time() - elapsed

            if METADATA_KEYARG in kwargs:
                result, sampleset_metadata = result

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
            if return_metadata:
                samplesets_data.append(sampleset_metadata)

            if save_results:
                np.save(f"{saving_path}_modularities", modularities)
                np.save(f"{saving_path}_communities", communities)
                if elapse_times:
                    np.save(f"{saving_path}_times", times)
                if return_metadata:
                    # Pickle saving tends to be safer for big objects
                    with open(f"{saving_path}_samplesets_data.pkl", "wb") as f:
                        pickle.dump(samplesets_data, f)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        if elapse_times and return_metadata and sampleset_metadata:
            return communities, modularities, times, sampleset_metadata
        if elapse_times:
            return communities, modularities, times
        if return_metadata:
            return communities, modularities, samplesets_data
        return communities, modularities

    def run_with_sampleset_info(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        iterative_verbosity: int = 0,
        return_metadata: bool = True,
        **kwargs,
    ):

        if return_metadata and not self.sampler.return_metadata:
            raise MethodArgsWarning(
                f"Set Advantage sampler's {METADATA_KEYARG} flag to True before running."
                + f" HierarchicalIterativeSearcher with {METADATA_KEYARG}."
            )

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))
        division_modularities = np.empty((num_runs), dtype=object)
        division_trees = np.empty((num_runs), dtype=object)
        # List instead of samplesets_data = np.empty((num_runs), dtype=object)
        # To prevent jupyter notebook kernel crashes
        # as handling big objects is not efficient with numpy dtype=object arrs
        samplesets_data = []

        if return_metadata:
            kwargs[METADATA_KEYARG] = True

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(
                return_modularities=True,
                division_tree=True,
                **kwargs,
            )

            # Currently only AdvantageSampler among the hierarchical solvers
            # provides sampleset metadata.
            if (
                isinstance(self.sampler, AdvantageSampler)
                and self.sampler.return_metadata
                and return_metadata
            ):
                (
                    communities_result,
                    div_tree,
                    div_modularities,
                    sampleset_data,
                ) = result
            else:
                (
                    communities_result,
                    div_tree,
                    div_modularities,
                ) = result
            times[iter] = time() - elapsed
            division_trees[iter] = div_tree
            division_modularities[iter] = div_modularities
            if return_metadata:
                # Pickle saving tends to be safer for big objects
                # and np.save does not support dtype=object
                samplesets_data.append(sampleset_data)

            try:
                modularity_score = nx.community.modularity(
                    self.searcher.sampler.G,
                    communities_result,
                    resolution=self.sampler.resolution,
                )
            except Exception as e:
                print(f"iteration: {iter} exception: {e}")
                modularity_score = -1

            communities[iter] = communities_result
            modularities[iter] = modularity_score

            if save_results:
                np.save(f"{saving_path}_modularities", modularities)
                np.save(f"{saving_path}_communities", communities)
                np.save(f"{saving_path}_times", times)
                np.save(f"{saving_path}_division_trees", division_trees)
                np.save(
                    f"{saving_path}_division_modularities",
                    division_modularities,
                )
                # Pickle saving tends to be safer for big objects
                if return_metadata:
                    with open(f"{saving_path}_samplesets_data.pkl", "wb") as f:
                        pickle.dump(samplesets_data, f)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        dtypes = [
            ("communities", object),
            ("modularity", np.float_),
            ("time", np.float_),
            ("division_tree", object),
            ("division_modularities", object),
        ]
        sampleset_components = [
            communities,
            modularities,
            times,
            division_trees,
            division_modularities,
        ]

        if return_metadata:
            dtypes.append(("samplesets_data", object))
            sampleset_components.append(samplesets_data)

        sampleset = np.rec.fromarrays(
            sampleset_components,
            dtype=dtypes,
        )

        return sampleset
