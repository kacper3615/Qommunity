import networkx as nx
import graph_tool as gt
from graph_tool.all import *
import numpy as np
from ..regular_sampler import RegularSampler
from ...utils import communities_to_dict, from_networkx_to_graphtool


class SBMSampler(RegularSampler):
    def __init__(self, G: nx.Graph, state_model=gt.inference.BlockState, state_args={"deg_corr": True, "entropy_args": {"degree_dl_kind": "distributed"}}, epropweight=True):
        self.G = G
        self.communities_number = None
        self.resolution = 1 # For compatibility 

        self.state_model = state_model
        self.state_args = state_args
        self.epropweight = epropweight

    def sample_qubo_to_dict(self) -> dict:
        communities = self.__SBMblocks()
        self.communities_number = len(communities)
        result = communities_to_dict(communities)
        return result

    def sample_qubo_to_list(self) -> list:
        communities = self.__SBMblocks()
        self.communities_number = len(communities)
        communities = list(map(list, communities))
        return communities

    def __SBMblocks(self):
        gtG, eprop_weight = from_networkx_to_graphtool(self.G)
        if self.epropweight:
            self.state_args["eweight"] = eprop_weight
    
        state = gt.inference.minimize_blockmodel_dl(gtG, self.state_model, state_args=self.state_args)
        
        node_assignment = np.array(state.get_blocks().a)
        communities = [{i for i,n in enumerate(node_assignment) if n==c} for c in np.unique(node_assignment)]
        
        return communities