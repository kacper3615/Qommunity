import numpy as np
import networkx as nx
from collections import defaultdict


def get_value(dictionary: dict) -> list:
    result = []
    for i in range(len(dictionary)):
        key = f"x{i}" 
        result.append(dictionary.get(key, None))
    return result

def calculate_modularity_matrix(graph, res) -> np.ndarray:
        adj_matrix: np.ndarray = nx.to_numpy_array(graph)
        degree_matrix: np.ndarray = adj_matrix.sum(axis=1)
        m: int = np.sum(degree_matrix)
        return (
            adj_matrix
            - res * np.outer(degree_matrix, degree_matrix) / m
        )

def calculate_generalized_modularity_matrix(graph, community, res=1) -> np.ndarray:
        if not community:
            community = [*range(graph.number_of_nodes())]
        else:
            community = list(community)

        full_B = calculate_modularity_matrix(graph, res)

        B_bis = full_B[community,:]
        B_community = B_bis[:,community]
        B_i = np.sum(B_community, axis=-1)
        delta = np.eye(len(community), dtype=np.int32)
        B_g = B_community - delta*B_i

        return B_g

def set_automatic_objective_function(B, degrees, resolution, community):
    Q = defaultdict(int)
    for i in range(len(B)):
        for j in range(len(B)):
            c_i, c_j = f"x{community[i]}", f"x{community[j]}"
            Q[(c_i, c_j)] = -B[i, j]
            if i == j:
                Q[(c_i, c_j)] += ((1 - resolution) * 
                                np.array(degrees, dtype=np.float32)[i])

    return Q