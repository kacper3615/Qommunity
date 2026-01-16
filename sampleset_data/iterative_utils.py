from dataclasses import dataclass, field
import networkx as nx
import numpy as np
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
    community_hashes: np.ndarray[list[str]] = field(init=False)

    cbfs: np.ndarray[object] = field(init=False)

    def __post_init__(self):
        cbfs = np.empty((len(self.hierarchical_metadatas),), dtype=object)
        for i, iterative_run in enumerate(self.hierarchical_metadatas):
            cbfs[i] = np.array([hm.chain_break_fraction for hm in iterative_run])
        self.cbfs = cbfs

        self._compute_community_hashes()

    def _compute_community_hashes(self):
        self.community_hashes = np.empty((len(self.hierarchical_metadatas),), dtype=object)
        for i, iterative_run in enumerate(self.communities):
            self.community_hashes[i] = hash(tuple(map(tuple, iterative_run)))

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
        cbfs_hierarchical_indexes_of_max = [
            np.argmax(cbfs_in_run) for cbfs_in_run in self.cbfs
        ]
        return cbfs_hierarchical_indexes_of_max

    @property
    def cbfs_hierarchical_indexes_of_means(self) -> np.ndarray[int]:
        cbfs_hierarchical_indexes_of_mean = [
            np.argmax(cbfs_in_run) for cbfs_in_run in self.cbfs
        ]
        return cbfs_hierarchical_indexes_of_mean
