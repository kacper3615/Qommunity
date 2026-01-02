import numpy as np
import networkx as nx

from dataclasses import dataclass, field
from saving_utils import parse_response_json, deserialize_qubo_problem
from searchers.utils import HierarchicalRunMetadata

@dataclass
class IterativeSearchGraphResults:
    graph_ref: nx.Graph
    graph_size: int
    communities: np.ndarray[list[list[int]]]
    modularities: np.ndarray[float]
    times: np.ndarray[float]
    division_modularities: np.ndarray[list[float]]
    division_trees: np.ndarray[list[list[int]]]
    hierarchical_metadatas: list[HierarchicalRunMetadata]

    cbfs: np.ndarray[object] = field(init=False)

    def __post_init__(self):
        cbfs = np.empty((len(self.hierarchical_metadatas),), dtype=object)
        for i, iterative_run in enumerate(self.hierarchical_metadatas):
            cbfs[i] = np.array([hm.chain_break_fraction for hm in iterative_run])
        self.cbfs = cbfs

    @property
    def chain_break_fractions(self) -> np.ndarray[float]:
        all_cbfs = np.concatenate(self.cbfs)
        return all_cbfs


    # CBFS maxes and means per hierarchical run
    @property
    def cbfs_maxes(self) -> np.ndarray[float]:
        cbfs_maxes = [np.max(cbfs_in_run) for cbfs_in_run in self.cbfs]
        return cbfs_maxes
    
    @property
    def cbfs_means(self) -> np.ndarray[float]:
        cbfs_means = [np.mean(cbfs_in_run) for cbfs_in_run in self.cbfs]
        return cbfs_means
    
    @property
    def cbfs_sum_per_run(self) -> np.ndarray[float]:
        cbfs_sums = [np.sum(cbfs_in_run) for cbfs_in_run in self.cbfs]
        return cbfs_sums
    
    # CBFS indexes of maxes and means in hierarchical runs
    @property
    def cbfs_hierarchical_indexes_of_max(self) -> np.ndarray[int]:
        cbfs_hierarchical_indexes_of_max = [np.argmax(cbfs_in_run) for cbfs_in_run in self.cbfs]
        return cbfs_hierarchical_indexes_of_max
    
    @property
    def cbfs_hierarchical_indexes_of_means(self) -> np.ndarray[int]:
        cbfs_hierarchical_indexes_of_mean = [np.argmax(cbfs_in_run) for cbfs_in_run in self.cbfs]
        return cbfs_hierarchical_indexes_of_mean
    
    # @property
    # def problem_ids(self) -> np.ndarray[str]:
    #     problem_ids = np.array([hm.problem_id for hm in self.hierarchical_metadatas])
    #     return problem_ids

    @property
    def problem_ids(self) -> np.ndarray[str]:
        problem_ids = np.empty((self.__len__() ), dtype=object)
        for iter in range(self.__len__()):
            problem_ids[iter] = self.hierarchical_metadatas[iter].problem_id
        return problem_ids

    @property
    def qpu_access_times_us(self) -> np.ndarray[float]:
        qpu_access_times = np.array([hm.dwave_sampleset_metadata.qpu_access_time_us for hm in self.hierarchical_metadatas])
        return qpu_access_times
    
    # @property
    # def problem(self) -> float:

    def __len__(self) -> int:
        return len(self.hierarchical_metadatas)

