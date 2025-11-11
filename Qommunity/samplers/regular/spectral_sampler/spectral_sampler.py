import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict


class SpectralSampler(RegularSampler):
    def __init__(self, G: nx.Graph, use_weights: bool = True, assign_labels: str = "cluster_qr", maxK: int = None):
        self.G = G
        self.communities_number = None
        self.weight = "weight" if use_weights else None
        self.resolution = 1 # For compatibility

        # See scikit-learn docs
        self.assign_labels = assign_labels 
        # Resolution limit for gamma=1
        self.maxK = int(np.sqrt(G.number_of_nodes())+1) if maxK is None else maxK 

    def sample_qubo_to_dict(self) -> dict:
        communities = self.__spectral_communities()
        self.communities_number = len(communities)
        result = communities_to_dict(communities)
        return result

    def sample_qubo_to_list(self) -> list:
        communities = self.__spectral_communities()
        self.communities_number = len(communities)
        communities = list(map(list, communities))
        return communities

    def __spectral_communities(self):
        A = nx.to_numpy_array(self.G, weight=self.weight)
        A = 0.5 *(A + A.T) # Safety if G is directed
        Q_spectral = 0
        communities = []

        for n_clusters in range(1, self.maxK):
            sc = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                assign_labels=self.assign_labels,
                random_state=None
            )
            labels = sc.fit_predict(A)
            spectral_comms = [set(np.where(labels == i)[0]) for i in range(n_clusters)]
            q = nx.community.modularity(self.G, spectral_comms, resolution=self.resolution)
            if q > Q_spectral:
                Q_spectral = q
                communities = spectral_comms
        
        return communities