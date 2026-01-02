from dataclasses import dataclass, field
import networkx as nx
import numpy as np

from searchers.utils import HierarchicalRunMetadata


@dataclass
class IterativeSearchForestResults:
    graphs_refs: np.ndarray[nx.Graph]
    graph_sizes: np.ndarray[int]
    communities: np.ndarray[np.ndarray[list[list[int]]]]
    modularities: np.ndarray[np.ndarray[float]]
    times: np.ndarray[np.ndarray[float]]
    division_modularities: np.ndarray[np.ndarray[list[float]]]
    division_trees: np.ndarray[np.ndarray[list[list[int]]]]
    hierarchical_metadatas: np.ndarray[list[HierarchicalRunMetadata]]

    cbfs: np.ndarray[object] = field(init=False)

    def __post_init__(self):
        cbfs = np.empty((len(self.hierarchical_metadatas),), dtype=object)
        for i, hierarchical_runs in enumerate(self.hierarchical_metadatas):
            cbfs[i] = np.array([hm.chain_break_fraction for hm in hierarchical_runs])
        self.cbfs = cbfs

