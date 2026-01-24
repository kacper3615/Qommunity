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
from Qommunity.searchers.utils import HierarchicalRunMetadata

from joblib import Parallel, delayed

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
        
    import numpy as np
    import networkx as nx
    from time import time
    from tqdm import tqdm
    from joblib import Parallel, delayed

    # Opcjonalne: zwiększa możliwości serializacji skomplikowanych obiektów
    try:
        import dill
        from joblib.externals.loky import set_loky_pickler
        set_loky_pickler('dill')
    except ImportError:
        pass

    def _single_iteration_worker(
        self,
        iter_idx, 
        searcher, 
        sampler, 
        saving_path, 
        save_results, 
        return_metadata,
        run_label,
        save_embeddings,
        kwargs
    ):
        """
        Funkcja pomocnicza wykonująca pojedynczy przebieg w osobnym procesie.
        """
        # run_label = f"iter_{iter_idx}"
        start_time = time()
        
        # 1. Wywołanie wyszukiwania
        # Uwaga: searcher musi być serializowalny przez pickle/dill
        result = searcher.hierarchical_community_search(
            return_modularities=True,
            division_tree=True,
            saving_path=saving_path,
            label=run_label,
            **kwargs,
        )

        # 2. Obsługa wyników zależnie od metadanych
        sampleset_data_single = None
        if return_metadata:
            communities_res, div_tree, div_mods, sampleset_data_single = result
        else:
            communities_res, div_tree, div_mods = result

        elapsed = time() - start_time

        # 3. Obliczenie modularności końcowej
        try:
            modularity_score = nx.community.modularity(
                sampler.G,
                communities_res,
                resolution=sampler.resolution,
            )
        except Exception as e:
            modularity_score = -1

        # 4. Zapisywanie metadanych dla tej konkretnej iteracji (jeśli dotyczy)
        if save_results and return_metadata and sampleset_data_single is not None:
            try:
                sampleset_data_single.save_to_files(base_filename=f"{saving_path}_{run_label}", 
                                                    save_embeddings=save_embeddings)
            except Exception as e:
                print(f"\n[Error] Iteration {iter_idx} saving error: {e}")

        return {
            "iter_idx": iter_idx,
            "communities": communities_res,
            "modularity": modularity_score,
            "time": elapsed,
            "division_tree": div_tree,
            "division_modularities": div_mods,
            "sampleset_data": sampleset_data_single
        }


    # --- Fragment wewnątrz Twojej klasy ---

    def run_with_sampleset_info(
        self,
        num_runs: int,
        save_results: bool = True,
        saving_path: str | None = None,
        iterative_verbosity: int = 0,
        return_metadata: bool = True,
        n_jobs: int = -1,  # -1 wykorzystuje wszystkie procesory
        run_labels: list[str] | None = None,
        save_embeddings: bool = True,
        **kwargs,
    ):
        # Sprawdzenie wymagań samplera
        if return_metadata and hasattr(self.sampler, "return_metadata") and not self.sampler.return_metadata:
            print("Warning: Metadata requested but sampler.return_metadata is False.")

        if iterative_verbosity >= 1:
            print(f"Starting parallel community detection: {num_runs} runs on {n_jobs} cores.")

        if save_results and saving_path is None:
            saving_path = self._default_saving_path()
        
        if return_metadata and isinstance(self.sampler, AdvantageSampler):
            kwargs[METADATA_KEYARG] = True
        else:
            return_metadata = False
            kwargs[METADATA_KEYARG] = False

        run_labels = run_labels if run_labels else [f"iter_{i}" for i in range(num_runs)]

        # --- RÓWNOLEGŁA PĘTLA Z TQDM ---
        # backend="loky" jest domyślny i najbezpieczniejszy dla NumPy
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self._single_iteration_worker)(
                i, 
                self.searcher, 
                self.sampler, 
                saving_path, 
                save_results, 
                return_metadata,
                run_label=f"{run_labels[i]}",
                save_embeddings=save_embeddings,
                kwargs=kwargs
            ) for i in tqdm(range(num_runs))
        )

        results = sorted(results, key=lambda r: r["iter_idx"])


        # --- REDUKCJA WYNIKÓW (Zbieranie danych z procesów) ---
        communities = np.array([r["communities"] for r in results], dtype=object)
        modularities = np.array([r["modularity"] for r in results], dtype=np.float64)
        times = np.array([r["time"] for r in results], dtype=np.float64)
        division_trees = np.array([r["division_tree"] for r in results], dtype=object)
        division_modularities = np.array([r["division_modularities"] for r in results], dtype=object)

        if save_results:
            np.save(f"{saving_path}_modularities", modularities)
            np.save(f"{saving_path}_communities", communities)
            np.save(f"{saving_path}_times", times)
            np.save(f"{saving_path}_division_trees", division_trees)
            np.save(f"{saving_path}_division_modularities", division_modularities)

        # Przygotowanie struktury rekordowej (np.recarray)
        dtypes = [
            ("communities", object),
            ("modularity", np.float64),
            ("time", np.float64),
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
            samplesets_data = np.array([r["sampleset_data"] for r in results], dtype=object)
            dtypes.append(("samplesets_data", object))
            sampleset_components.append(samplesets_data)
            
        sampleset = np.rec.fromarrays(
            sampleset_components,
            dtype=dtypes,
        )

        return sampleset

    # def run_with_sampleset_info(
    #     self,
    #     num_runs: int,
    #     save_results: bool = True,
    #     saving_path: str | None = None,
    #     iterative_verbosity: int = 0,
    #     return_metadata: bool = True,
    #     **kwargs,
    # ):

    #     if return_metadata and hasattr(self.sampler, "return_metadata") and not self.sampler.return_metadata:
    #         raise MethodArgsWarning(
    #             f"Set Advantage sampler's {METADATA_KEYARG} flag to True before running."
    #             + f" HierarchicalIterativeSearcher with {METADATA_KEYARG}."
    #         )

    #     if iterative_verbosity >= 1:
    #         print("Starting community detection iterations")

    #     if save_results and saving_path is None:
    #         saving_path = self._default_saving_path()

    #     modularities = np.zeros((num_runs))
    #     communities = np.empty((num_runs), dtype=object)
    #     times = np.zeros((num_runs))
    #     division_modularities = np.empty((num_runs), dtype=object)
    #     division_trees = np.empty((num_runs), dtype=object)
    #     samplesets_data = np.empty((num_runs), dtype=object)

    #     if return_metadata and isinstance(self.sampler, AdvantageSampler):
    #         kwargs[METADATA_KEYARG] = True
    #     else:
    #         return_metadata = False
    #         kwargs[METADATA_KEYARG] = False

    #     for iter in tqdm(range(num_runs)):
    #         run_label = f"iter_{iter}"
            
    #         elapsed = time()
    #         result = self.searcher.hierarchical_community_search(
    #             return_modularities=True,
    #             division_tree=True,
    #             saving_path=saving_path,
    #             label=run_label,
    #             **kwargs,
    #         )

    #         # Currently only AdvantageSampler among the hierarchical solvers
    #         # provides sampleset metadata.
    #         if (
    #             isinstance(self.sampler, AdvantageSampler)
    #             and self.sampler.return_metadata
    #             and return_metadata
    #         ):
    #             (
    #                 communities_result,
    #                 div_tree,
    #                 div_modularities,
    #                 sampleset_data,
    #             ) = result
    #         else:
    #             (
    #                 communities_result,
    #                 div_tree,
    #                 div_modularities,
    #             ) = result
    #         times[iter] = time() - elapsed
    #         division_trees[iter] = div_tree
    #         division_modularities[iter] = div_modularities
    #         if return_metadata:
    #             samplesets_data[iter] = sampleset_data

    #         try:
    #             modularity_score = nx.community.modularity(
    #                 self.searcher.sampler.G,
    #                 communities_result,
    #                 resolution=self.sampler.resolution,
    #             )
    #         except Exception as e:
    #             print(f"iteration: {iter} exception: {e}")
    #             modularity_score = -1

    #         communities[iter] = communities_result
    #         modularities[iter] = modularity_score

    #         if save_results:
    #             np.save(f"{saving_path}_modularities", modularities)
    #             np.save(f"{saving_path}_communities", communities)
    #             np.save(f"{saving_path}_times", times)
    #             np.save(f"{saving_path}_division_trees", division_trees)
    #             np.save(
    #                 f"{saving_path}_division_modularities",
    #                 division_modularities,
    #             )
    #             # Pickle saving tends to be safer for big objects
    #             if return_metadata:
    #                 try:
    #                     sampleset_data.save_to_files(base_filename=f"{saving_path}_{run_label}")
    #                 except Exception as e:
    #                     print(f"Error while saving HierarchicalRunMetadata (sampleset_data) from iteration: {iter}", e)
                    

    #         if iterative_verbosity >= 1:
    #             print(f"Iteration {iter} completed")

    #     dtypes = [
    #         ("communities", object),
    #         ("modularity", np.float64),
    #         ("time", np.float64),
    #         ("division_tree", object),
    #         ("division_modularities", object),
    #     ]
    #     sampleset_components = [
    #         communities,
    #         modularities,
    #         times,
    #         division_trees,
    #         division_modularities,
    #     ]

    #     if return_metadata:
    #         dtypes.append(("samplesets_data", object))
    #         sampleset_components.append(samplesets_data)
            
    #     sampleset = np.rec.fromarrays(
    #         sampleset_components,
    #         dtype=dtypes,
    #     )

    #     return sampleset
