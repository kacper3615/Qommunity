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

    def run(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        elapse_times: bool = True,
        iterative_verbosity: int = 0,
        return_sampleset_info: bool = False,
        **kwargs,
    ):
        kwargs = self._verify_kwargs(kwargs)

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))
        sampleset_infos = np.empty((num_runs), dtype=object)

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(**kwargs)
            times[iter] = time() - elapsed

            if "return_sampleset_info" in kwargs:
                result, sampleset_info = result

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
            sampleset_infos[iter] = sampleset_info

            if save_results:
                np.save(f"{saving_path}_modularities", modularities)
                np.save(f"{saving_path}_communities", communities)
                if elapse_times:
                    np.save(f"{saving_path}_times", times)
                if return_sampleset_info:
                    np.save(f"{saving_path}_sampleset_infos", sampleset_infos)

            if iterative_verbosity >= 1:
                print(f"Iteration {iter} completed")

        if elapse_times and sampleset_info:
            return communities, modularities, times, sampleset_info
        if elapse_times:
            return communities, modularities, times
        if return_sampleset_info:
            return communities, modularities, sampleset_infos
        return communities, modularities

    def run_with_sampleset_info(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        iterative_verbosity: int = 0,
        return_sampleset_info: bool = True,
        process_results: bool = True,
        **kwargs,
    ):

        if iterative_verbosity >= 1:
            print("Starting community detection iterations")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()

        modularities = np.zeros((num_runs))
        communities = np.empty((num_runs), dtype=object)
        times = np.zeros((num_runs))
        division_modularities = np.empty((num_runs), dtype=object)
        division_trees = np.empty((num_runs), dtype=object)
        sampleset_infos = np.empty((num_runs), dtype=object)

        if return_sampleset_info:
            kwargs["return_sampleset_info"] = True

        for iter in tqdm(range(num_runs)):
            elapsed = time()
            result = self.searcher.hierarchical_community_search(
                return_modularities=True,
                division_tree=True,
                **kwargs,
            )
            if return_sampleset_info:
                (
                    communities_result,
                    div_tree,
                    div_modularities,
                    sampleset_info,
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
            sampleset_infos[iter] = sampleset_info

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
                np.save(f"{saving_path}_sampleset_infos", sampleset_infos)

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

        if return_sampleset_info:
            dtypes.append(("sampleset_infos", object))
            sampleset_components.append(sampleset_infos)

        sampleset = np.rec.fromarrays(
            sampleset_components,
            dtype=dtypes,
        )

        if not process_results:
            return sampleset

        dtype = [si.dwave_sampleset_info for si in sampleset[0].sampleset_infos][
            0
        ].dtype.descr
        dwave_sampleset_infos = np.array(
            [
                np.concatenate(
                    [
                        np.array([r], dtype=dtype)
                        for r in [
                            si.dwave_sampleset_info
                            for si in sampleset[run].sampleset_infos
                        ]
                    ]
                ).view(np.recarray)
                for run in range(len(sampleset))
            ],
            dtype=object,
        )

        dtype = [si.time_measurements for si in sampleset[0].sampleset_infos][
            0
        ].dtype.descr
        time_measurements = np.array(
            [
                np.concatenate(
                    [
                        np.array([r], dtype=dtype)
                        for r in [
                            si.time_measurements
                            for si in sampleset[run].sampleset_infos
                        ]
                    ]
                ).view(np.recarray)
                for run in range(len(sampleset))
            ],
            dtype=object,
        )

        results_procesed_dtypes = sampleset.dtype.descr
        results_procesed_dtypes.pop()
        results_procesed_dtypes.append(("dwave_sampleset_infos", object))
        results_procesed_dtypes.append(("time_measurements", object))
        results_procesed_dtypes

        results_processed_componenets = [
            sampleset.communities,
            sampleset.modularity,
            sampleset.time,
            sampleset.division_tree,
            sampleset.division_modularities,
            dwave_sampleset_infos,
            time_measurements,
        ]

        results_processed = np.rec.fromarrays(
            results_processed_componenets,
            dtype=results_procesed_dtypes,
        )

        return results_processed
