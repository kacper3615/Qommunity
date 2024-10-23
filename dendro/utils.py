import seaborn as sns
import matplotlib.pyplot as plt


def nodes_to_communities(communities: list) -> dict:
    result = {}
    for i in range(len(communities)):
        for node in communities[i]:
            result[node] = i
    return result


def get_autoscaled_colormap(n_clusters, communities):
    if n_clusters <= 10:
        return sns.color_palette("hsv", len(communities))
    elif n_clusters <= 20:
        cmap = plt.get_cmap("tab20", n_clusters)
        return [cmap(i % 10) for i in range(len(communities))]
    else:
        cmap = plt.get_cmap("hsv", n_clusters)
        return [cmap(i % 10) for i in range(len(communities))]


# TODO
def autoscale_fig_width(nodes: list) -> float:
    num_nodes = len(nodes)
    base_width = 10
    node_factor = 0.4
    width = base_width + num_nodes * node_factor
    max_width = 100
    return min(width, max_width)


# TODO
def autoscale_fig_height(nodes: list) -> float:
    num_nodes = len(nodes)
    base_height = 5
    height_factor = 0.3
    height = base_height + num_nodes * height_factor
    max_height = 30
    return min(height, max_height)
