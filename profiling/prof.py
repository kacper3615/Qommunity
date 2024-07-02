from tqdm import tqdm
import numpy as np
import networkx as nx
import os

from Qommunity.samplers.hierarchical.advantage_sampler import AdvantageSampler
from Qommunity.searchers.hierarchical_community_searcher import HierarchicalCommunitySearcher



def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)



MIN_NODES = 10
MAX_NODES = 100

num_nodes = np.linspace(MIN_NODES, MAX_NODES, MAX_NODES//MIN_NODES)


dir = "networks/powerlaw_m=1_p=0.2"
os.makedirs(os.path.dirname(dir+"/"), exist_ok=True)

solver = "advantage"


try:
    Graphs = np.load(f"{dir}/graphs.npy", allow_pickle=True)
except:
    Graphs = np.empty(shape=(len(num_nodes),), dtype=object)
    for i, n in enumerate(num_nodes):
        Graphs[i] = nx.powerlaw_cluster_graph(n=n, m=1, p=0.2)
    np.save(f"{dir}/graphs.npy", Graphs)




dir = "output"
os.makedirs(os.path.dirname(dir+"/"), exist_ok=True)
make_dir(f"{dir}/{solver}")


N_RUNS = 1

big_graph_idx = MAX_NODES//MIN_NODES - 1


mods = np.zeros((num_nodes[big_graph_idx:].shape[0], N_RUNS))
comms = np.empty((num_nodes[big_graph_idx:].shape[0], N_RUNS), dtype=object)

mods_graph_N = np.zeros((N_RUNS))
comms_graph_N = np.empty((N_RUNS), dtype=object)
for i, G in tqdm(enumerate(Graphs[big_graph_idx:])):
    advantage = AdvantageSampler(G, num_reads=100)
    hierch_searcher = HierarchicalCommunitySearcher(advantage)
    
    # current_net_size = int(num_nodes[i])
    current_net_size = int(G.number_of_nodes())
    for r in range(N_RUNS):
        comms_res = hierch_searcher.hierarchical_community_search()
        mod_score = nx.community.modularity(G, comms_res)
        mods_graph_N[r] = mod_score
        comms_graph_N[r] = comms_res

        # For bigger graphs computations take time, so better save it up
        # even for each run
        try:
            np.save(f"{dir}/{solver}/{solver}-network_size_{current_net_size}", mods_graph_N)
            np.save(f"{dir}/{solver}/{solver}-network_size_{current_net_size}_comms", comms_graph_N)
        except Exception as e:
            print(f"iter {i} run {r} npy saving failed:\n{e}")

    mods[i] = mods_graph_N
    comms[i] = comms_graph_N
    np.save(f"{dir}/{solver}/{solver}-network_size_{current_net_size}", mods_graph_N)
    np.save(f"{dir}/{solver}/{solver}-network_size_{current_net_size}_comms", comms_graph_N)

np.save(f"{dir}/{solver}/{solver}-mods_final", mods)
np.save(f"{dir}/{solver}/{solver}-comms_final", comms)